#![feature(substr_range)]
#![allow(clippy::cognitive_complexity, clippy::cast_precision_loss)]
#![warn(clippy::unwrap_in_result, clippy::unwrap_used)]
#![allow(clippy::cognitive_complexity)]
#![warn(
    rust_2018_idioms,
    rust_2021_compatibility,
    arithmetic_overflow,
    nonstandard_style,
    clippy::disallowed_types,
    clippy::nursery,
    // clippy::pedantic
)]
use std::{
    cell::OnceCell,
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    fs,
    hash::{BuildHasherDefault, DefaultHasher, Hasher},
    io::{BufRead, BufReader, BufWriter, Cursor, Read, Seek, SeekFrom, Write},
    mem::offset_of,
    ops::ControlFlow,
    path::{Path, PathBuf},
    sync::{atomic::AtomicUsize, Arc, LazyLock, OnceLock, RwLock},
    time::{Duration, Instant},
};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use clap::{CommandFactory, Parser, Subcommand, ValueEnum};
use color_eyre::eyre::{self, bail, Result};
use common::{BeamWidth, RelativeTime, System, SystemFlags, TreeNode};
use crossbeam_channel::{bounded, unbounded};
use data_loader::HEADER_SIZE;
use directories::ProjectDirs;
use fs_err::File;
use human_repr::{HumanCount, HumanDuration, HumanThroughput};
use itertools::Itertools;
use memmap2::Mmap;
use nohash_hasher::NoHashHasher;
use num_traits::ToBytes;
use parse_display::IntoResult;
use rand::prelude::*;
use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelRefIterator, ParallelBridge,
        ParallelIterator,
    },
    prelude::IntoParallelIterator,
    slice::{ParallelSlice, ParallelSliceMut},
};
use roaring::{MultiOps, RoaringBitmap};
use serde::{Deserialize, Serialize};
use shadow_rs::shadow;
use sif_kdtree::{KdTree, WithinDistance};
use termimad::MadSkin;
use tracing::*;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};
use triple_accel::levenshtein::levenshtein_simd_k_str;
use yansi::Paint;

// use yansi::{Condition, Paint};
use crate::{
    common::{
        AstronavError, AstronavResult, FormatFloat, FormatNum, SysEntry, F32,
    },
    route::{ModeConfig, Router},
    ship::NamedShip,
};

mod common;
mod data_loader;
mod event;
mod galaxy;
#[cfg(feature = "gui_eframe")]
mod gui_eframe;
#[cfg(feature = "gui_iced")]
mod gui_iced;
mod journal;
mod route;
mod ship;

shadow!(build);

// #[global_allocator]
// static GLOBAL: dhat::Alloc = dhat::Alloc;

#[derive(Deserialize, Serialize, Debug)]
struct Job {
    stars: Option<PathBuf>,
    ship: Option<String>,
    hops: JobHops,
    mode: ModeConfig,
    range: Option<f32>,
    #[serde(default)]
    deterministic: bool,
    #[serde(default)]
    quiet: bool,
    #[serde(default)]
    mmap: bool,
}

#[derive(Deserialize, Serialize, Debug)]
#[serde(untagged)]
enum JobHops {
    File(PathBuf),
    List(Vec<String>),
}

impl JobHops {
    fn resolve(self, stars: &Path) -> Result<Vec<u32>> {
        match self {
            Self::File(file) => {
                let lines = BufReader::new(File::open(file)?)
                    .lines()
                    .map(|l| match l {
                        Ok(s) => Ok(s.trim().to_owned()),
                        Err(e) => Err(e.into()),
                    })
                    .collect::<Result<Vec<String>>>()?;
                Self::List(lines).resolve(stars)
            }
            Self::List(list) => {
                let mut r = Router::default();
                r.set_path(stars)?;
                let mut entries = Vec::with_capacity(list.len());
                for entry in list {
                    entries.push(entry.parse()?);
                }
                let systems = r.resolve(&entries)?;
                let mut ret = Vec::with_capacity(entries.len());
                for (entry, sys) in entries.into_iter().zip(systems.iter()) {
                    if let Some(sys) = sys {
                        print_sys_match(entry, sys);
                        ret.push(sys.id);
                    } else {
                        return Err(
                            AstronavError::SystemNotFoundError(entry).into()
                        );
                    }
                }
                Ok(ret)
            }
        }
    }
}

impl Job {
    fn run(self) -> Result<Vec<System>> {
        let stars = self.stars.unwrap_or_else(|| data_dir().join("stars"));
        let mut router = Router::default();
        let hops = self.hops.resolve(&stars)?;
        router.set_path(stars)?;
        if let Some(ship) = self.ship {
            router.set_ship(get_ship(&ship)?)
        }
        let (dt, route) = router
            .compute_route(&hops, self.range, 0.0, self.mode, self.mmap)?;
        print_route(dt, &route, self.quiet);
        Ok(route)
    }
}

#[derive(ValueEnum, Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case", untagged)]
enum RefuelMode {
    /// Refuel when possible
    WhenPossible,
    /// Refuel when fuel level is below max fuel per jump
    WhenEmpty,
    /// Same as above, but only refuel to have enough to make the next jump
    LeastJumps,
}

#[derive(ValueEnum, Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case", untagged)]
enum ShipMode {
    /// Minimize amount of fuel spent
    Fuel,
    /// Minimize number of jumps
    Jumps,
}

#[derive(Subcommand, Debug, Deserialize, Serialize, Clone)]
enum Mode {
    /// Beam search
    #[clap(visible_aliases(["beam"]))]
    BeamSearch {
        /// Beam width in the form of <width> (absolute), <den/num>
        /// (fractional), "inf"/"infinite"
        #[arg(short, long)]
        beam_width: Option<BeamWidth>,
        /// Refueling mode to use
        #[arg(short, long, value_enum)]
        refuel_mode: Option<RefuelMode>,
        /// Only refuel on primary/arrival stars
        #[arg(long)]
        refuel_primary: bool,
        /// Only use primary/arrival stars for FSD boost
        #[arg(long)]
        boost_primary: bool,
        /// Percentage of maximum jump range to use to conserve fuel
        #[arg(long)]
        range_limit: Option<f32>,
    },
    /// Greedy deptht first search (very fast, but results in very long routes)
    DepthFirst,
    /// Incremntally broadening beam search (not yet implemented)
    IncrementalBroadening,
    /// Beam Stack Search (not yet implemented)
    BeamStack,
    /// Incremental Beam Search
    #[clap(visible_aliases(["ibs"]))]
    IncrementalBeamSearch {
        #[arg(default_value_t = 1024)]
        beam_width: usize,
    },
    /// A* search with adjustable weight (results in routes with low fuel
    /// consumption)
    #[clap(visible_aliases(["astar"]))]
    AStar {
        /// Adjust weight of heuristic (0=Dijkstra,1.0=AStar,inf=Greedy)
        #[arg(short, long, default_value_t = 1.0)]
        weight: f32,
    },
    /// Dijkstra's shortest path algorithm
    Dijkstra,
    /// Take fuel consumption, refueling and system reachability into account,
    /// *very* slow and incomplete
    Ship {
        /// What to optimize for, least fuel consumed or least amount of jumps
        #[arg(short, long, value_enum)]
        mode: ShipMode,
    },
}

fn data_dir() -> PathBuf {
    let Some(path) = ProjectDirs::from("", "", env!("CARGO_PKG_NAME")) else {
        panic!("Could not get project folder!")
    };
    path.data_local_dir().to_owned()
}

#[derive(Subcommand, Debug, Clone)]
enum Command {
    #[cfg(any(feature = "gui_iced", feature = "gui_eframe"))]
    /// Launch gui
    Gui,
    /// Test
    Test { key: String },
    /// Preprocess Spansh's galaxy.json.gz dump
    PreprocessGalaxy {
        /// Name of data files to write (foo/stars will write
        /// foo/stars.dat,foo/stars.idx, foo/stars.names)
        stars_path: PathBuf,
        /// Path of the galaxy dump to process (if not given download dump from
        /// spansh)
        galaxy_path: Option<PathBuf>,
    },
    /// Computer route betweem star systems
    Route {
        /// Name of data files to load (foo/stars will load
        /// foo/stars.dat,foo/stars.idx,foo/stars.name)
        #[arg(long,short='p',default_value=data_dir().join("stars").into_os_string())]
        stars_path: PathBuf,
        /// Name of ship loadout to load from journal
        #[arg(long, short)]
        ship: Option<String>,
        /// Jump range to use if no ship is specified
        #[arg(long, short)]
        range: Option<f32>,
        /// Make route computation fully deterministic even when running with
        /// multiple threads
        #[arg(long, short)]
        deterministic: bool,
        /// Reduce memory usage by memory mapping the KD-Tree used for route
        /// computation
        #[arg(long = "mmap", short)]
        mmap_tree: bool,
        /// Reduce memory usage by only loading star system within a maximum
        /// distance from the straight line connecting route waypoints
        #[arg(long, short)]
        line_dist: Option<f32>,
        /// Don't print computed route when done (mostly useful for benchmarks
        /// and testing)
        #[arg(long, short)]
        quiet: bool,
        /// Routing algorithm to use
        #[command(subcommand)]
        mode: Mode,
        /// System names, ":" prefixed Astronav IDs, "#" prefixed ID64s or
        /// positions in the form of x/y/z to compute route between
        #[clap(global = true)]
        hops: Vec<SysEntry>,
    },
}

fn long_version() -> String {
    use std::fmt::Write;
    let mut ret = version();
    ret.push('\n');
    if !build::TAG.is_empty() {
        writeln!(&mut ret, "tag: {}", build::TAG)
            .unwrap_or_else(|_| unreachable!());
    }
    writeln!(&mut ret, "rustc {},{}", build::RUST_VERSION, build::RUST_CHANNEL)
        .unwrap_or_else(|_| unreachable!());
    ret
}

fn version() -> String {
    format!(
        "v{} ({} {})",
        build::PKG_VERSION,
        build::SHORT_COMMIT,
        build::BUILD_TIME
    )
}

#[derive(Parser, Debug)]
#[command(author, about, long_about = None, version = version(), long_version = long_version())]
struct Cli {
    /// Show full help
    #[arg(long)]
    help_full: bool,
    /// Minimum time between printing progress updates
    #[arg(short, long, default_value_t = 0.5)]
    progress_interval: f64,
    /// Number of threads to use
    #[arg(short, long, default_value_t = num_cpus::get())]
    threads: usize,
    /// Command to run
    #[command(subcommand)]
    cmd: Option<Command>,
}

impl TryFrom<Mode> for ModeConfig {
    type Error = AstronavError;
    fn try_from(val: Mode) -> AstronavResult<Self> {
        let mode = match val {
            Mode::BeamSearch {
                beam_width,
                refuel_mode,
                refuel_primary,
                boost_primary,
                range_limit,
            } => {
                let beam_width = beam_width.unwrap_or_default();
                let range_limit = range_limit.unwrap_or_default();
                let refuel_mode = match refuel_mode {
                    Some(RefuelMode::LeastJumps) => {
                        Some(route::RefuelMode::LeastJumps)
                    }
                    Some(RefuelMode::WhenEmpty) => {
                        Some(route::RefuelMode::WhenEmpty)
                    }
                    Some(RefuelMode::WhenPossible) => {
                        Some(route::RefuelMode::WhenPossible)
                    }
                    None => None,
                };
                Self::BeamSearch {
                    beam_width,
                    refuel_mode,
                    refuel_primary,
                    boost_primary,
                    range_limit,
                }
            }
            Mode::AStar { weight } => Self::AStar { weight: F32(weight) },
            Mode::Ship { mode } => {
                let ship_mode = match mode {
                    ShipMode::Fuel => route::ShipMode::Fuel,
                    ShipMode::Jumps => route::ShipMode::Jumps,
                };
                Self::Ship { ship_mode }
            }
            Mode::DepthFirst => Self::DepthFirst,
            Mode::IncrementalBroadening => Self::IncrementalBroadening,
            Mode::BeamStack => Self::BeamStack,
            Mode::Dijkstra => Self::Dijkstra,
            Mode::IncrementalBeamSearch { beam_width } => {
                Self::IncrementalBeamSearch { beam_width }
            }
        };
        Ok(mode)
    }
}

fn get_ship(name: &str) -> AstronavResult<NamedShip> {
    let ships = ship::Ship::new_from_journal()?;
    let mut best = (u32::MAX, None);
    for ship in &ships {
        if ship.ident == name || ship.ship_name.as_deref() == Some(name) {
            best.1 = Some((format!("{ship}"), ship));
            best.0 = 0;
            break;
        }
        let ship_name = format!("{ship}");
        let k: Result<u32, _> = ship_name.len().max(name.len()).try_into();
        let k = k.map_err(|e| AstronavError::Other(e.into()))?;
        if let Some(d) = levenshtein_simd_k_str(name, &ship_name, k) {
            if d < best.0 {
                best.1 = Some((ship_name, ship));
                best.0 = d;
            }
        }
    }
    if let Some((_, ship)) = best.1 {
        return Ok(ship.clone());
    }
    Err(AstronavError::RuntimeError(format!(
        "Ship {name:?} not found, available ship: {ship_names}",
        ship_names = ships.iter().map(|s| format!("{s}")).join(", ")
    )))
}

fn print_sys_match(sys_entry: SysEntry, sys: &System) {
    use yansi::{Color, Paint};
    if let SysEntry::Name(name) = &sys_entry {
        if name == &sys.name {
            info!("{sys} [EXACT MATCH]", sys = sys.green());
        } else {
            info!("{name} => {sys}", name = name.yellow(), sys = sys.green(),);
        }
    } else {
        info!("{sys_entry} => {sys}");
    }
}

fn print_route(dt: Duration, route: &[System], quiet: bool) {
    use yansi::{Color::*, Paint, Style};
    let route_dist: f32 = route
        .windows(2)
        .map(|w| match w {
            [a, b] => a.distp(b),
            _ => 0.0,
        })
        .sum();
    info!(
        "Route computed in {}: {} jumps, {} Ly",
        dt.human_duration(),
        route.len().format_num(),
        route_dist.format_float()
    );
    if quiet {
        return;
    }
    println!(
        "Star Types: r=Regular, s=Scoopable, w=White Dwarf, n=Neutron Star, f=Refueling stop, uppercase for arrival star, lowercase for other"
    );
    for (n, hop) in route.iter().enumerate() {
        let dist = if n < route.len() - 1 {
            route[n].distp(&route[n + 1])
        } else {
            0.0
        };
        let mut style = match (hop.flags.primary_kind(), hop.flags.kind()) {
            (common::StarKind::Scoopable, _) => Style::new().green().bright(),
            (_, common::StarKind::Scoopable) => Style::new().green().dim(),
            (common::StarKind::Neutron, _) => Style::new().blue().bright(),
            (_, common::StarKind::Neutron) => Style::new().blue().dim(),
            (common::StarKind::WhiteDwarf, _) => Style::new().cyan().bright(),
            (_, common::StarKind::WhiteDwarf) => Style::new().cyan().dim(),
            _ => Style::new(),
        };
        if hop.flags.is_waypoint() {
            style = style.underline();
        }
        if hop.flags.is_refuel() {
            style = style.bold().red();
            if hop.flags.primary_kind() == common::StarKind::Scoopable {
                style = style.bright();
            }
        }
        let tag = hop.tag();
        let tag = tag.paint(style);
        let name = hop.name.paint(style);
        if hop.flags.is_waypoint() {
            println!(
                "<{n}> {name} ({tag}) {dist}",
                dist = dist.human_count(" Ly")
            );
        } else {
            println!(
                " {n}  {name} ({tag}) {dist}",
                dist = dist.human_count(" Ly")
            );
        }
    }
}

fn init_tracing() {
    tracing_subscriber::registry()
        .with(
            fmt::layer()
                .event_format(fmt::format().with_ansi(yansi::is_enabled()))
                .with_timer(RelativeTime::default())
                .compact(),
        )
        .with(
            EnvFilter::try_from_env("ASTRONAV_LOG")
                .unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();
}

const D_MAX: f32 = 50.0;

fn sif_kdtree_bench(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
) -> Result<()> {
    use std::sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    };

    use criterion::BatchSize;
    use sif_kdtree::{Distance, KdTree, Object, WithinDistance};
    let (data, _) = data_loader::load(
        &r#"C:\Users\Earthnuker\AppData\Local\astronav\data\stars.bin"#,
        &[],
        0.0,
    )?;
    let t_start = Instant::now();
    let tree: Arc<KdTree<_, _>> = Arc::new(KdTree::par_new(data));
    println!("Built in {}", t_start.elapsed().human_duration());
    let total = tree.len();
    let seed = Arc::new(AtomicU64::new(0));
    group.throughput(criterion::Throughput::Elements(1 as u64));
    let id = criterion::BenchmarkId::from_parameter(format!("KD:par"));
    group.bench_with_input(id, &tree, move |b, tree| {
        b.iter_batched_ref(
            || {
                let mut rng = rand::rngs::StdRng::seed_from_u64(
                    seed.fetch_add(1, Ordering::SeqCst),
                );
                let idx = rng.gen_range(0..total);
                let n = tree[idx];
                (
                    Mutex::new(Vec::with_capacity(1024)),
                    WithinDistance::new(*n.pos(), D_MAX),
                )
            },
            |(buf, n)| {
                criterion::black_box(tree.par_look_up(n, |o| {
                    buf.lock().expect("Failed to lock buffer").push(o);
                    ControlFlow::Continue(())
                }));
            },
            BatchSize::SmallInput,
        );
    });
    Ok(())
}

fn sif_kdtree_serial_bench(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
) -> Result<()> {
    use std::sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    };

    use criterion::BatchSize;
    use sif_kdtree::{Distance, KdTree, Object, WithinDistance};
    let (data, _) = data_loader::load(
        &r#"C:\Users\Earthnuker\AppData\Local\astronav\data\stars.bin"#,
        &[],
        0.0,
    )?;
    let t_start = Instant::now();
    let tree: Arc<KdTree<_, _>> = Arc::new(KdTree::new(data));
    println!("Built in {}", t_start.elapsed().human_duration());
    let total = tree.len();
    let seed = Arc::new(AtomicU64::new(0));
    group.throughput(criterion::Throughput::Elements(1 as u64));
    let id = criterion::BenchmarkId::from_parameter(format!("KD:par"));
    group.bench_with_input(id, &tree, move |b, tree| {
        b.iter_batched_ref(
            || {
                let mut rng = rand::rngs::StdRng::seed_from_u64(
                    seed.fetch_add(1, Ordering::SeqCst),
                );
                let idx = rng.gen_range(0..total);
                let n = tree[idx];
                (Vec::with_capacity(1024), WithinDistance::new(*n.pos(), D_MAX))
            },
            |(buf, n)| {
                criterion::black_box(tree.look_up(n, |o| {
                    buf.push(o);
                    ControlFlow::Continue(())
                }));
            },
            BatchSize::SmallInput,
        );
    });
    Ok(())
}

mod bench {
    use std::time::Duration;

    use criterion::{
        black_box, criterion_group, criterion_main, measurement::WallTime,
        BenchmarkGroup, BenchmarkId, Criterion,
    };

    use super::Result;
    use crate::{common::TreeNode, route::Router};

    fn bench_tree(c: &mut Criterion) {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_cpus::get())
            .build_global()
            .unwrap();
        let mut group = c.benchmark_group("bench_tree");
        // super::rstar_bench(&mut group).unwrap();
        super::sif_kdtree_bench(&mut group).unwrap();
        super::sif_kdtree_serial_bench(&mut group).unwrap();
        // super::sif_rtree_bench(&mut group, 512).unwrap();
        // super::benchmark_router::<LargeNodeParameters<2, 0, 0>>(&mut group);
        // super::benchmark_router::<LargeNodeParameters<4, 0, 0>>(&mut group);
        // super::benchmark_router::<LargeNodeParameters<4, 1, 0>>(&mut group);
        // super::benchmark_router::<LargeNodeParameters<8, 0, 0>>(&mut group);
        // super::benchmark_router::<LargeNodeParameters<8, 1, 0>>(&mut group);
        // super::benchmark_router::<LargeNodeParameters<16, 0, 0>>(&mut group);
        // super::benchmark_router::<LargeNodeParameters<16, 1, 0>>(&mut group);
        // super::benchmark_router::<LargeNodeParameters<32, 0, 0>>(&mut group);
        // super::benchmark_router::<LargeNodeParameters<32, 1, 0>>(&mut group);
        // super::benchmark_router::<LargeNodeParameters<64, 0, 0>>(&mut group);
        // super::benchmark_router::<LargeNodeParameters<64, 1, 0>>(&mut group);
        // super::benchmark_router::<LargeNodeParameters<128, 0, 0>>(&mut
        // group); super::benchmark_router::<LargeNodeParameters<128, 1,
        // 0>>(&mut group);
        // super::benchmark_router::<LargeNodeParameters<256, 0, 0>>(&mut
        // group); super::benchmark_router::<LargeNodeParameters<256, 1,
        // 0>>(&mut group);
        // super::benchmark_router::<LargeNodeParameters<512, 0, 0>>(&mut
        // group); super::benchmark_router::<LargeNodeParameters<512, 1,
        // 0>>(&mut group);
        // super::benchmark_router::<LargeNodeParameters<1024, 0, 0>>(&mut
        // group); super::benchmark_router::<LargeNodeParameters<1024,
        // 1, 0>>(&mut group);
    }

    criterion_group!(benches, bench_tree);
    criterion_main!(benches);
    pub fn run() {
        main();
    }
}

fn main() -> Result<()> {
    // let _profiler = dhat::Profiler::new_heap();
    use yansi::Condition;
    color_eyre::install()?;
    yansi::whenever(Condition::DEFAULT);
    run()?;
    // drop(_profiler);
    Ok(())
}

fn full_help() {
    let docs = clap_markdown::help_markdown::<Cli>();
    let skin = MadSkin::no_style();
    eprintln!("{}", skin.term_text(&docs));
}

// 5467B0DA1D106495
fn grid_test() -> Result<()> {
    use std::{
        fs::File,
        io::{BufRead, Read, Write},
    };
    const SECTOR_SIZE: f32 = 1280.0;
    let path =
        PathBuf::from(r"C:\Users\Earthnuker\AppData\Local\astronav\data\stars");
    let (mapped_nodes, _) = data_loader::map(&path)?;
    let mut grid: BTreeMap<[i16; 3], BufWriter<File>> = BTreeMap::default();
    let mut base_dir = PathBuf::new();
    base_dir.push("bins");
    std::fs::remove_dir_all(&base_dir);
    let tree = sif_kdtree::KdTree::new_unchecked(mapped_nodes);
    for (id, node) in tree.iter().enumerate() {
        let mut hasher = fnv::FnvHasher::with_key(0);
        let pos = node.pos();
        let mut cube_id = [0i16; 3];
        for (i, p) in pos.iter().enumerate() {
            cube_id[i] = ((p / SECTOR_SIZE).trunc() as i64).try_into()?;
            hasher.write_i16(cube_id[i]);
        }
        let key = hasher.finish().to_be_bytes();
        let fh = grid.entry(cube_id).or_insert_with(|| {
            let folder = format!("{:02X}", key[0]);
            let key: String =
                key[1..].iter().map(|b| format!("{b:02X}")).collect();
            let mut path = base_dir.clone();
            path.push(folder);
            std::fs::create_dir_all(&path).unwrap();
            path.push(key);
            let mut wr = BufWriter::new(File::create(&path).unwrap());
            wr.write_i16::<LittleEndian>(cube_id[0]).unwrap();
            wr.write_i16::<LittleEndian>(cube_id[1]).unwrap();
            wr.write_i16::<LittleEndian>(cube_id[2]).unwrap();
            wr
        });
        fh.write_u32::<LittleEndian>(id.try_into()?)?;
        fh.write_u8(node.flags.value())?;
        for p in pos {
            fh.write_f32::<LittleEndian>(*p)?;
        }
    }
    drop(tree);
    let sector_size = SECTOR_SIZE as i64;
    // TODO: single file, multiple MMaps
    for (k, mut v) in grid.into_iter() {
        let k = [
            k[0] as i64 * sector_size,
            k[1] as i64 * sector_size,
            k[2] as i64 * sector_size,
        ];
        let a = [
            (k[0] - (sector_size / 2)) as f32,
            (k[1] - (sector_size / 2)) as f32,
            (k[2] - (sector_size / 2)) as f32,
        ];
        let b = [
            (k[0] + (sector_size / 2)) as f32,
            (k[1] + (sector_size / 2)) as f32,
            (k[2] + (sector_size / 2)) as f32,
        ];
        v.flush()?;
        let pos = v.get_ref().stream_position()?;
        let mut v = v.into_inner()?;
        println!("{a:?}-{b:?}: {pos}");
        let (fh, mm, slice) = unsafe {
            let x = v.read_u32::<LittleEndian>()?;
            let y = v.read_u32::<LittleEndian>()?;
            let z = v.read_u32::<LittleEndian>()?;
            let mm = Mmap::map(&v)?;
            let total = (mm.len() - (3 * 4)) / (4 + 1 + 3 * 4);
            let slice = std::slice::from_raw_parts(mm[3 * 4..].as_ptr(), total);
            (v, mm, slice)
        };
    }
    let t_start = Instant::now();
    println!(
        "Done in {dt}! sleeping infinitely!",
        dt = t_start.elapsed().human_duration()
    );
    loop {
        std::thread::sleep(Duration::from_secs(1));
    }
    Ok(())
}

fn run() -> Result<()> {
    let args = std::env::args().collect_vec();
    if args.len() == 2 && args[1].starts_with('%') {
        init_tracing();
        let job: Job = toml::from_str(&fs::read_to_string(&args[1][1..])?)?;
        job.run()?;
        std::process::exit(0);
    }
    let args = Cli::parse_from(argfile::expand_args(
        argfile::parse_fromfile,
        argfile::PREFIX,
    )?);

    if args.help_full {
        full_help();
        return Ok(());
    }

    #[cfg(any(feature = "gui_iced", feature = "gui_eframe"))]
    if args.cmd.is_none() {
        args.cmd = Some(Command::Gui);
    }

    #[cfg(any(feature = "gui_iced", feature = "gui_eframe"))]
    if !matches!(args.cmd, Some(Command::Gui)) {
        init_tracing();
    }
    #[cfg(not(any(feature = "gui_iced", feature = "gui_eframe")))]
    init_tracing();

    let (ctrl_c_tx, ctrl_c_rx) = bounded(0);
    // println!("{args:#?}");
    ctrlc::set_handler(move || {
        if let Err(e) = ctrl_c_tx.send(()) {
            error!("Failed to send Ctrl-C: {e}");
        };
    })?;
    let mut router =
        Router::new(Duration::from_secs_f64(args.progress_interval));
    {
        let ctrl_c_rx = ctrl_c_rx.clone();
        router.set_callback(Box::new(move |router, event| {
            if ctrl_c_rx.try_recv().is_ok() {
                router.interrupt();
            }
            info!("{event}");
            Ok(())
        }));
    }
    let Some(cmd) = args.cmd else {
        Cli::command().print_long_help()?;
        return Ok(());
    };
    rayon::ThreadPoolBuilder::new().num_threads(args.threads).build_global()?;
    match cmd {
        #[cfg(any(feature = "gui_iced", feature = "gui_eframe"))]
        Command::Gui => {
            #[cfg(feature = "gui_eframe")]
            return gui_eframe::main(router);
            #[cfg(feature = "gui_iced")]
            return gui_iced::main(router).map_err(Into::into);
        }
        Command::Test { ref key } if key.as_str() == "mmap" => {
            // galaxy::make_tree(r"C:\Users\Earthnuker\AppData\Local\astronav\
            // data\stars2")?;
        }
        Command::Test { ref key } if key.as_str() == "optimize" => {
            let mut mode = ModeConfig::BeamSearch {
                beam_width: BeamWidth::Infinite,
                refuel_mode: None,
                refuel_primary: true,
                boost_primary: true,
                range_limit: 0.0,
            };
            // Setup
            let mut last_state = Arc::new(RwLock::new(None));
            {
                let ctrl_c_rx = ctrl_c_rx.clone();
                let last_state = last_state.clone();
                router.set_callback(Box::new(move |router, event| {
                    if ctrl_c_rx.try_recv().is_ok() {
                        error!("Exit requested!");
                        std::process::exit(1);
                    }
                    info!("{event}");
                    if let event::Event::SearchState(route_state) = event {
                        let mut state = last_state.write().unwrap();
                        *state = Some(route_state.clone());
                    }
                    Ok(())
                }));
            }
            // ==========

            router.set_path(
                &r#"C:\Users\Earthnuker\AppData\Local\astronav\data\stars.bin"#,
            )?;
            router.load(true, &[], 0.0)?;
            let mut bws = vec![];
            (0..17).for_each(|n| bws.push(BeamWidth::Absolute(1 << n)));
            bws.push(BeamWidth::Infinite);
            bws = vec![BeamWidth::Absolute(8192)];
            // let mut fh = BufWriter::new(File::create("opt_log.csv")?);
            // info!("Ready!");
            // let mut stdin = std::io::stdin();
            // let _ = stdin.read(&mut [0u8]).unwrap();
            let tree = router.get_tree();
            let total = tree.len();
            let mut stats = vec![];
            let waypoints = &[
                // DW2 Waypoints
                "Pallaeni",
                "Omega Sector VE-Q b5-15",
                "Pru Aescs NC-M d7-192",
                "Clooku EW-Y c3-197",
                "Boewnst KS-S c20-959",
                "Dryau Ausms KG-Y e3390",
                "Stuemeae KM-W c1-342",
                "Hypiae Phyloi LR-C d22",
                "Phroi Bluae QI-T e3-3454",
                "Bleethuae NI-B d674",
                "Smootoae QY-S d3-202",
                "Beagle Point",
                // Additional waypoints
                "Colonia",
                "Sol",
            ]
            .into_iter()
            .map(|s| SysEntry::Name(s.to_owned()))
            .collect::<Vec<SysEntry>>();
            let hops = router
                .resolve(&waypoints)?
                .into_iter()
                .map(|s| s.unwrap())
                .collect::<Vec<_>>();
            let hop_ids = hops.iter().map(|h| h.id).collect::<Vec<_>>();
            let range = 50.0;
            for bw in &bws {
                if let ModeConfig::BeamSearch { beam_width, .. } = &mut mode {
                    *beam_width = *bw;
                }
                last_state.write().unwrap().take();
                let route_res = router.compute_route(
                    &hop_ids,
                    Some(range),
                    0.0,
                    mode,
                    true,
                );
                let last_state = last_state.write().unwrap().take();
                let (dt, route) = match route_res {
                    Ok((dt, route)) => (dt, route),
                    Err(e) => {
                        error!("{e}");
                        stats.push((
                            range,
                            bw,
                            vec![],
                            f32::NAN,
                            last_state,
                            Duration::from_secs_f32(0.0),
                        ));
                        continue;
                    }
                };
                let route_dist: f32 = route
                    .iter()
                    .tuple_windows()
                    .map(|(a, b)| common::dist(&a.pos, &b.pos))
                    .sum();
                let goal_dist: f32 = hops
                    .iter()
                    .tuple_windows()
                    .map(|(a, b)| common::dist(&a.pos, &b.pos))
                    .sum();
                // writeln!(
                //     &mut fh,
                //     "{bw} @ {range:.02} | {src} -> {dst} |
                // {goal_dist:.02} Ly ({route_dist:.02} Ly) | Jumps: {l} |
                // Visited: {v:.02}% ",
                //     src = src_sys.name,
                //     dst = dst_sys.name,
                //     l = route.len(),
                //     v = last_state.prc_seen
                // )?;
                // fh.flush()?;
                stats.push((range, bw, route, route_dist, last_state, dt));
            }
            let mut data = vec![];
            for (range, bw, route, route_dist, last_state, dt) in stats {
                data.push(serde_json::json!({
                    "bw": bw,
                    "range": range,
                    "dist": route_dist,
                    "jumps": route.len(),
                    "visited": last_state.as_ref().map(|s| s.prc_seen).unwrap_or(0.0),
                    "dt": dt.as_secs_f64(),
                    "hops": hops,
                }));
                info!(
                    "[{bw} @ {range:.02} Ly] Dist: {dr} Ly, Jumps: {l}, visited: {v:.02}%, {dt}",
                    dr = route_dist,
                    l = route.len(),
                    v = last_state.as_ref().map(|s| s.prc_seen).unwrap_or(0.0),
                    dt = dt.human_duration()
                );
            }
            eprintln!("{}", serde_json::to_string(&data)?);
        }
        Command::Test { ref key } if key.as_str() == "grid_test" => {
            grid_test()?;
        }
        Command::Test { ref key } if key.as_str() == "beam_efficiency" => {
            use rustc_hash::FxHashMap;
            let mut rng = thread_rng();
            router.set_path(
                r#"C:\Users\Earthnuker\AppData\Local\astronav\data\stars.bin"#,
            )?;
            router.load(true, &[], 0.0)?;
            router.set_ship(get_ship("NMGR")?);
            let tree = router.get_tree();
            let tree = tree.as_ref();
            let mut acc_speedup = FxHashMap::default();
            let mut acc_diff = FxHashMap::default();
            let mut results: BTreeMap<BeamWidth, Vec<(usize, Duration, f32)>> =
                BTreeMap::default();
            'outer: loop {
                let hops = tree.choose_multiple(&mut rng, 2).collect_vec();
                let ids = hops.iter().map(|hop| tree.id(hop)).collect_vec();
                let systems = ids.iter().map(|&id| router.get(id).expect("Failed to get system")).collect_vec();
                let route_dist = systems
                    .iter()
                    .tuple_windows()
                    .map(|(a, b)| common::dist(&a.pos, &b.pos))
                    .sum::<f32>();
                println!(
                    "{ids:?}: {route_dist:.02} Ly",
                    route_dist = route_dist
                );
                let mut res = BTreeMap::new();
                for bw in 0usize..=14 {
                    let bw = match bw {
                        0 => BeamWidth::Infinite,
                        1..7 => {
                            continue;
                        }
                        n => BeamWidth::Absolute(1 << (n - 1)),
                    };
                    let route = router.compute_route(
                        &ids,
                        None,
                        0.0,
                        ModeConfig::BeamSearch {
                            beam_width: bw,
                            refuel_mode: None,
                            refuel_primary: true,
                            boost_primary: true,
                            range_limit: 0.0,
                        },
                        true,
                    );
                    let (dt,route) = match route {
                        Ok((dt, route)) => (dt, route),
                        Err(e) => {
                            error!("{e}");
                            continue 'outer;
                        }
                    };
                    let dist = route
                        .iter()
                        .tuple_windows()
                        .map(|(a, b)| common::dist(&a.pos, &b.pos))
                        .sum::<f32>();
                    res.insert(bw, (route.len(), dt, dist));
                }
                let Some((b_len, b_dt, _)) = res.get(&BeamWidth::Infinite)
                else {
                    continue;
                };
                for (bw, (jumps, dt, dist)) in &res {
                    let speedup = b_dt.as_secs_f64() / dt.as_secs_f64();
                    *acc_speedup.entry(*bw).or_insert(0.0) += speedup;
                    let route_diff = *b_len as i64 - *jumps as i64;
                    let route_diff_rel = route_diff as f64 / *b_len as f64;
                    *acc_diff.entry(*bw).or_insert(0.0) += route_diff_rel;
                    println!(
                        "{bw}: {jumps} jumps, {dt}, {dist:.02} Ly",
                        bw = bw,
                        jumps = jumps,
                        dt = dt.human_duration(),
                        dist = dist
                    );
                    println!(
                        "Speedup: {:.02}, Diff: {}",
                        b_dt.as_secs_f64() / dt.as_secs_f64(),
                        (*jumps as i64)-(*b_len as i64)
                    );
                    results.entry(*bw).or_default().push((*jumps, *dt, *dist));
                }
                let Some(baseline) = results.get(&BeamWidth::Infinite) else {
                    continue;
                };
                for (bw, res) in &results {
                    if res.len() != baseline.len() {
                        warn!("Skipping {bw} due to length mismatch");
                        continue;
                    }
                    let mean_speedup = baseline
                        .iter()
                        .zip(res)
                        .map(|(b, r)| b.1.as_secs_f64() / r.1.as_secs_f64())
                        .sum::<f64>()
                        / res.len() as f64;
                    let mean_diff = baseline
                        .iter()
                        .zip(res)
                        .map(|(b, r)| {
                            (r.0 as i64 - b.0 as i64) as f64
                        })
                        .sum::<f64>()
                        / res.len() as f64;
                    println!("{bw}: {mean_speedup:.02}, {mean_diff:.02}");
                }
                // let speedup = dt_inf.as_secs_f64() / dt_beam.as_secs_f64();
                // let route_diff = route_beam.len() as i64 - route_inf.len() as
                // i64; let route_diff_rel = route_diff as f64 /
                // route_inf.len() as f64; acc_speedup +=
                // speedup; acc_diff += route_diff_rel;
                // loops += 1.0;
                // println!(
                //     "{sys_0} -> {sys_1}: {route_dist:.02} Ly, Beam: {dt_beam}
                // ({dist_beam:.02}), Inf: {dt_inf} ({dist_inf:02}), Speedup:
                // {speedup:.02}, Diff: {route_diff} ({route_diff_rel:.02}%,
                // {dist_diff:.02} Ly)",     sys_0=sys_0,
                //     sys_1=sys_1,
                //     route_dist=route_dist,
                //     dist_beam=dist_beam.human_count("Ly"),
                //     dist_inf=dist_inf.human_count("Ly"),
                //     dt_beam=dt_beam.human_duration(),
                //     dt_inf=dt_inf.human_duration(),
                //     speedup=speedup,
                //     route_diff=route_diff,
                //     route_diff_rel=route_diff_rel*100.0,
                //     dist_diff = dist_beam-dist_inf,
                // );
                // println!(
                //     "Speedup: {:.02}, Diff: {:.02} %", acc_speedup/loops,
                // (acc_diff*100.0)/loops );
            }
        }
        Command::Test { ref key } if key.as_str() == "rayon" => {
            router.set_path(
                &r#"C:\Users\Earthnuker\AppData\Local\astronav\data\stars.bin"#,
            )?;
            router.load(true, &[], 0.0)?;
            let tree = router.get_tree();
            let router = Arc::new(router);
            let (tx, rx) = bounded::<(u32, u64)>(1024 * 1024);
            let mut t = std::thread::spawn(move || {
                tree.par_iter().chunks(4096).for_each(|chunk| {
                    // dbg!(rayon::current_num_threads(),
                    // rayon::current_thread_index());
                    chunk.into_par_iter().for_each(|node| {
                        let tx = tx.clone();
                        let mut n = 0;
                        tree.look_up(
                            &WithinDistance::new(
                                *node.pos(),
                                node.mult() * 50.0,
                            ),
                            |nb| {
                                n += 1;
                                ControlFlow::Continue(())
                            },
                        );
                        tx.send((tree.id(node), n)).unwrap();
                    });
                });
            });
            let mut count = BTreeMap::new();
            let mut n = 0u64;
            rx.into_iter().for_each(|(k, v)| {
                let e = count.entry(k).or_insert(0);
                *e += v;
                n += v;
                if (n % 1_000_000) == 0 {
                    let keys = count.len() as u64;
                    let max = count.values().max().unwrap();
                    let min = count.values().min().unwrap();
                    let mean = n / keys;
                    println!(
                        "{keys}: {n}, (Min: {min}, Max: {max}, Mean: {mean})"
                    );
                }
            });
            t.join().unwrap();
            let keys = count.len() as u64;
            let max = count.values().max().unwrap();
            let min = count.values().min().unwrap();
            let mean = n / keys;
            println!("{keys}: {n}, (Min: {min}, Max: {max}, Mean: {mean})");
        }

        Command::Test { ref key } if key.as_str() == "close" => {
            // router.set_path(
            //     &r#"C:\Users\Earthnuker\AppData\Local\astronav\data\stars.
            // bin"#, )?;
            // router.load(&[], 0.0)?;
            // let tree = router.get_tree();
            // let mut best = (f32::INFINITY,0xffffffffu32,0xffffffffu32);
            // let mut worst = (f32::NEG_INFINITY,0xffffffffu32,0xffffffffu32);
            // for entry in tree.iter() {
            //     let p =
            // tree.nearest_neighbor_iter(&entry.pos).nth(2).unwrap();
            //     let d = common::dist(&entry.pos,&p.pos);
            //     if d<best.0 && d>0.0 {
            //         best=(d,entry.id,p.id);
            //         println!("[B] {best:?}");
            //     }
            //     if d>worst.0 {
            //         worst=(d,entry.id,p.id);
            //         println!("[W] {worst:?}");
            //     }
            // }
            // let best_0 = router.get(best.1)?;
            // let best_1 = router.get(best.2)?;
            // let worst_0 = router.get(worst.1)?;
            // let worst_1 = router.get(worst.2)?;
            // println!("Best: [{best_0}] - [{best_1}]: {d}",
            // d=best.0.human_count("Ly")); println!("Worst:
            // [{worst_0}] - [{worst_1}]:  {d}",d=worst.0.human_count("Ly"));
        }

        Command::Test { ref key } if key.as_str() == "nbs" => {
            router.set_path(
                &r#"C:\Users\Earthnuker\AppData\Local\astronav\data\stars.bin"#,
            )?;
            router.load(true, &[], 0.0)?;
            let tree = router.get_tree();
            let total = tree.len();
            let mut maps = tree
                .iter()
                .enumerate()
                .par_bridge()
                .map(|(n, node)| {
                    if n % 1000 == 0 {
                        let prc = ((n as f64) / (total as f64)) * 100.0;
                        info!("Processing {n}/{total} ({prc:.2}%)")
                    }
                    let mut bm = HashSet::new();
                    tree.look_up(
                        &WithinDistance::new(*node.pos(), node.mult() * 48.0),
                        |n| {
                            bm.insert(tree.id(n));
                            ControlFlow::Continue(())
                        },
                    );
                    let bm = (tree.id(node), bm);
                    bm
                })
                .collect_vec_list()
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();
            maps.par_sort_unstable_by_key(|(n, _)| *n);
            let mut fh = BufWriter::new(File::create("nbs.bm")?);
            let mut sizes = vec![0];
            for (n, bm) in maps {
                if n % 10000 == 0 {
                    let prc = ((n as f64) / (total as f64)) * 100.0;
                    info!("Writing {n}/{total} ({prc:.2}%)")
                }
                let bm = RoaringBitmap::from_sorted_iter(bm.into_iter())?;
                sizes.push(bm.serialized_size());
                bm.serialize_into(&mut fh)?;
            }
            // let systems = router
            //     .resolve(&[
            //         SysEntry::Name("Sol".to_owned()),
            //         SysEntry::Name("Colonia".to_owned()),
            //     ])?
            //     .into_iter()
            //     .map(|s| s.unwrap())
            //     .collect::<Vec<_>>();
            // let start_sys = systems.get(0).unwrap();
            // let goal_sys = systems.get(1).unwrap();
            // let route = router.route_mcts(
            //     &start_sys,
            //     &goal_sys,
            //     48.0,
            //     &BeamWidth::Absolute(512),
            //     None,
            //     true,
            //     true,
            //     1.0,
            // )?;
            // router.precomp_bfs(48.0)?;
            // trigram_test()?;
            // index_test()?;
            // names_test(false)?;
            // eudex_test()?;
            // grid_test()?;
            return Ok(());
        }
        Command::Test { ref key } => {
            bail!("Unknown test command: {key}");
        }
        Command::PreprocessGalaxy { stars_path, galaxy_path } => {
            galaxy::process_galaxy_dump(
                galaxy_path.as_ref(),
                &stars_path,
                &router,
            )?;
        }
        Command::Route {
            ship,
            stars_path,
            mode,
            mut range,
            hops,
            deterministic,
            line_dist,
            quiet,
            mmap_tree,
        } => {
            use yansi::{Color::*, Paint, Style};
            let mode = mode.try_into()?;
            if let Some(ship_name) = ship {
                let ship = get_ship(&ship_name)?;
                range.get_or_insert(ship.range());
                router.set_ship(ship);
            }
            router.set_path(&stars_path)?;
            let systems = router.resolve(&hops)?;
            router.set_deterministic(deterministic);
            let hop_pos = systems
                .iter()
                .map(|s| s.as_ref().map(|s| s.pos))
                .collect::<Option<Vec<_>>>()
                .unwrap_or_default();
            let mut sys_ids = Vec::with_capacity(hops.len());
            for (sys_entry, sys) in hops.into_iter().zip(systems.iter()) {
                if let Some(sys) = sys {
                    print_sys_match(sys_entry, sys);
                    sys_ids.push(sys.id);
                } else {
                    return Err(
                        AstronavError::SystemNotFoundError(sys_entry).into()
                    );
                }
            }
            router.load(mmap_tree, &hop_pos, line_dist.unwrap_or(0.0))?;
            let (t_route, route) =
                router.compute_route(&sys_ids, range, 0.0, mode, mmap_tree)?;
            print_route(t_route, &route, quiet);
        }
    }
    Ok(())
}

// fn render_nodes(nodes: Vec<TreeNode>, px: &mut pixels::Pixels) -> Result<()>
// {     let (w, h) = {
//         let tx = px.texture();
//         (tx.width() as usize, tx.height() as usize)
//     };
//     Ok(())
// }

// fn render() -> Result<()> {
//     use common::StarKind;
//     use pixels::{Pixels, SurfaceTexture};
//     use winit::{
//         dpi::LogicalSize,
//         event::{Event, VirtualKeyCode},
//         event_loop::{ControlFlow, EventLoop},
//         window::WindowBuilder,
//     };
//     use winit_input_helper::WinitInputHelper;

//     const WIDTH: usize = 1024;
//     const HEIGHT: usize = 1024;

//     let event_loop = EventLoop::new();
//     let mut input = WinitInputHelper::new();
//     let window = {
//         let size = LogicalSize::new(WIDTH as f64, HEIGHT as f64);
//         let scaled_size = LogicalSize::new(WIDTH as f64, HEIGHT as f64);
//         WindowBuilder::new()
//             .with_title("Galaxy Viewer")
//             .with_inner_size(scaled_size)
//             .with_min_inner_size(size)
//             .build(&event_loop)
//             .expect("failed to create window")
//     };
//     let mut pixels = {
//         let window_size = window.inner_size();
//         let surface_texture =
//             SurfaceTexture::new(window_size.width, window_size.height,
// &window);         Pixels::new(
//             WIDTH.try_into().expect("Failed to convert"),
//             HEIGHT.try_into().expect("Failed to convert"),
//             surface_texture,
//         )?
//     };
//     let t_start = Instant::now();
//     let path = PathBuf::from(r"C:\Users\Earthnuker\astronav\data\stars");
//     let (nodes, _) = data_loader::load(&path, &[], 0.0)?;
//     let (p_min, p_max) = nodes
//         .par_chunks(4096)
//         .map(|chunk| {
//             let mut acc = ([f32::MAX; 3], [f32::MIN; 3]);
//             for v in chunk {
//                 acc = (
//                     [
//                         acc.0[0].min(v.pos[0]),
//                         acc.0[1].min(v.pos[1]),
//                         acc.0[2].min(v.pos[2]),
//                     ],
//                     [
//                         acc.1[0].max(v.pos[0]),
//                         acc.1[1].max(v.pos[1]),
//                         acc.1[2].max(v.pos[2]),
//                     ],
//                 );
//             }
//             acc
//         })
//         .collect::<Vec<_>>()
//         .into_iter()
//         .reduce(|acc, pos| {
//             (
//                 [
//                     acc.0[0].min(pos.0[0]),
//                     acc.0[1].min(pos.0[1]),
//                     acc.0[2].min(pos.0[2]),
//                 ],
//                 [
//                     acc.1[0].max(pos.1[0]),
//                     acc.1[1].max(pos.1[1]),
//                     acc.1[2].max(pos.1[2]),
//                 ],
//             )
//         })
//         .unwrap_or_default();

//     fn scale(
//         p: [f32; 3],
//         p_min: [f32; 3],
//         p_max: [f32; 3],
//         scale: f32,
//     ) -> [f32; 3] { let mut res = [ scale * (p[0] - p_min[0]) / (p_max[0] -
//       p_min[0]), scale * (p[1] - p_min[1]) / (p_max[1] - p_min[1]), scale *
//       (p[2] - p_min[2]) / (p_max[2] - p_min[2]), ]; // res[0]=scale-res[0];
//       res[2] = scale - res[2]; res
//     }
//     let pos: Vec<_> = nodes
//         .par_iter()
//         .map(|node| {
//             (
//                 node.flags,
//                 scale(node.pos, p_min, p_max, WIDTH.min(HEIGHT) as f32),
//             )
//         })
//         .collect();
//     let mut acc = vec![vec![0f64; WIDTH]; HEIGHT];
//     for (flags, p) in pos {
//         let (px, py) = pixels.clamp_pixel_pos((p[0] as _, p[2] as _));
//         acc[py][px] += flags.mult() as f64;
//     }
//     let max_v: f64 = acc
//         .par_iter()
//         .flatten()
//         .max_by(|a, b| a.total_cmp(b))
//         .copied()
//         .unwrap_or_default()
//         .log2();
//     let buffer: Vec<u8> = acc
//         .into_par_iter()
//         .flatten()
//         .flat_map(|v| {
//             let v = ((v.log2() / max_v) * 255.0) as u8;
//             [v, v, v, 0xff]
//         })
//         .collect();
//     pixels.frame_mut().copy_from_slice(&buffer[..]);
//     info!("Processing took {}", t_start.elapsed().human_duration());
//     event_loop.run(move |event, _, control_flow| {
//         if let Event::RedrawRequested(_) = event {
//             if let Err(err) = pixels.render() {
//                 error!("{}", err);
//                 *control_flow = ControlFlow::Exit;
//                 return;
//             }
//         }
//         if input.update(&event) {
//             if input.key_pressed(VirtualKeyCode::Escape)
//                 || input.close_requested()
//             {
//                 *control_flow = ControlFlow::Exit;
//             }
//             window.request_redraw();
//         }
//     });
//     Ok(())
// }

// fn do_test() -> Result<()> {
//     use std::sync::Arc;

//     use human_repr::*;
//     let db = sled::Config::new()
//         .path(r"C:\Users\Earthnuker\astronav\data\test.db")
//         .use_compression(false)
//         .compression_factor(22) // max: 22
//         .create_new(true)
//         // .cache_capacity(1024 * 1024 * 256) // 256 MB
//         .print_profile_on_drop(true)
//         .open()?;
//     let tree_sys = typed_sled::Tree::<u32,DbSystem>::open(&db, "systems");
//     let tree_id = typed_sled::Tree::<u64,u32>::open(&db, "id64");
//     let path = PathBuf::from(r"C:\Users\Earthnuker\astronav\data\stars");
//     let mut lc = Arc::into_inner(LineCache::new(&path)?)
//         .expect("unreachable")
//         .into_inner()?;
//     let (nodes, _) = data_loader::load(&path, &[], 0.0)?;
//     let t_start = Instant::now();
//     let total_nodes = nodes.len() as f64;
//     for (n, node) in nodes.into_iter().enumerate() {
//         let sys: System = lc.get(node.id)?;
//         let db_sys = sys.clone().into();
//         tree_id.insert(&sys.id64, &sys.id)?;
//         tree_sys.insert(&sys.id, &db_sys)?;
//         if n % 100_000 == 0 {
//             let prc = (n as f64 / total_nodes) * 100.0;
//             let t_rem = Duration::from_secs_f64({
//                 if n==0 {
//                     0.0
//                 } else {
//                     let dt = t_start.elapsed().as_secs_f64()/(n as f64);
//                     dt*(total_nodes-(n as f64))
//                 }
//             });
//             info!(
//                 "Wrote {n} entries ({prc:.02}% , {size}, {rem})",
//                 size = db
//                     .size_on_disk()
//                     .unwrap_or(0)
//                     .human_count_bytes(),
//                 rem=t_rem.human_duration()
//             );
//         }
//         if n % 1_000_000 == 0 {
//             tree_id.flush()?;
//         }
//     }
//     info!("Took: {}", t_start.elapsed().human_duration());
//     Ok(())
//     // render()?;
//     // Ok(())
// }

fn parse_record(r: &[u8; 8 + 4]) -> Result<u64> {
    let mut c = Cursor::new(r);
    let id64 = c.read_u64::<LittleEndian>()?;
    let pos = c.read_u32::<LittleEndian>()?;
    if c.position() != u64::try_from(r.len())? {
        bail!("Leftover data!");
    }
    Ok(id64)
}

#[derive(Debug)]
struct CompressedIndex {
    index: Vec<u32>,
    sizes: Vec<u32>,
    id64_hi: Vec<u32>,
    mm_idx: memmap2::Mmap,
    mm_names: memmap2::Mmap,
}

const fn split_id64(id: u64) -> (u32, u32) {
    let lo = (id & 0xffffffff) as u32;
    let id = id >> 32;
    let hi = (id & 0xffffffff) as u32;
    (hi, lo)
}

impl CompressedIndex {
    fn load<P: AsRef<Path>>(path: &P) -> Result<Self> {
        let path = path.as_ref();
        let mm_idx = unsafe { Mmap::map(&File::open(path)?) }?;
        let (sizes, index, id64_hi) = {
            let mut c = Cursor::new(&mm_idx[16..]);
            let size = c.read_u32::<LittleEndian>()?;
            let mut id64_hi = Vec::with_capacity(size as usize);
            let mut index = Vec::with_capacity(size as usize);
            let mut sizes = Vec::with_capacity(size as usize);
            for _ in 0..size {
                id64_hi.push(c.read_u32::<LittleEndian>()?);
                index.push(c.read_u32::<LittleEndian>()?);
                sizes.push(c.read_u32::<LittleEndian>()?);
            }
            (sizes, index, id64_hi)
        };
        let mm_names =
            unsafe { Mmap::map(&File::open(path.with_extension("names"))?) }?;
        Ok(Self { mm_idx, mm_names, index, sizes, id64_hi })
    }

    fn get_id64_by_id(&self, id: u32) -> Result<u64> {
        let mut start_id = 0;
        let mut chunk_index = 0;
        for (n, size) in self.sizes.iter().enumerate() {
            if start_id + size > id {
                chunk_index = n;
                break;
            }
            start_id += size;
        }
        assert!(id > start_id);
        let offset = self.index[chunk_index] as usize;
        let mut c = Cursor::new(&self.mm_idx[offset..]);
        let id64_hi = c.read_u32::<LittleEndian>()? as u64;
        c.read_u32::<LittleEndian>()?; // skip name_pos
        let count = c.read_u32::<LittleEndian>()?;
        let id64_lo = if count == 0 {
            roaring::RoaringBitmap::deserialize_from(&mut c)?
                .iter()
                .nth((id - start_id) as usize)
                .ok_or_else(|| eyre::anyhow!("System ID {id} not found"))?
                as u64
        } else {
            c.seek(std::io::SeekFrom::Current(
                i64::from(id) - i64::from(start_id),
            ))?;
            c.read_u32::<LittleEndian>()? as u64
        };
        Ok(id64_hi << 32 | id64_lo)
    }

    fn get_name_by_id(&self, id: u32) -> Result<String> {
        let mut start_id = 0;
        let mut chunk_index = 0;
        for (n, size) in self.sizes.iter().enumerate() {
            if start_id + size > id {
                chunk_index = n;
                break;
            }
            start_id += size;
        }
        let offset = self.index[chunk_index] as usize;
        let mut c = Cursor::new(&self.mm_idx[offset + 4..]);
        let names_pos = c.read_u32::<LittleEndian>()? as usize;
        let names = BufReader::new(Cursor::new(&self.mm_names[names_pos..]));
        names
            .lines()
            .map_while(Result::ok)
            .nth((id - start_id) as usize)
            .ok_or_else(|| eyre::anyhow!("System ID {id} not found!"))
    }

    fn get_by_id64(
        &self,
        id64: u64,
        with_name: bool,
    ) -> Result<(u32, Option<String>)> {
        info!("get_by_id64({id64})");
        let (hi, lo) = split_id64(id64);
        let Ok(index) = self.index.binary_search_by_key(&hi, |offset| {
            let mut c = Cursor::new(&self.mm_idx[(*offset as usize)..]);
            match c.read_u32::<LittleEndian>() {
                Ok(res) => res,
                Err(e) => panic!("Failed to read u32: {e}"),
            }
        }) else {
            bail!("ID {id64} not found!");
        };
        let offset = self.index[index] as usize;
        let chunk_start_id: u32 = self.sizes.iter().take(index).sum();
        let mut c = Cursor::new(&self.mm_idx[offset..]);
        let id_hi = c.read_u32::<LittleEndian>()?;
        assert_eq!(id_hi, hi);
        let names_pos = c.read_u32::<LittleEndian>()? as usize;
        let count = c.read_u32::<LittleEndian>()? as usize;
        let idx = if count == 0 {
            info!("ROARING!");
            let bitmap = roaring::RoaringBitmap::deserialize_from(&mut c)?;
            if !bitmap.contains(lo) {
                bail!("ID not found!");
            }
            usize::try_from(bitmap.rank(lo) - 1)?
        } else {
            info!("LIST!");
            let mut entries = vec![0u32; count];
            c.read_u32_into::<LittleEndian>(&mut entries)?;
            let Some(res) = entries.iter().position(|&v| v == lo) else {
                bail!("ID not found!");
            };
            res
        };
        let names = BufReader::new(Cursor::new(&self.mm_names[names_pos..]));
        let id = chunk_start_id + idx as u32;
        if !with_name {
            return Ok((id, None));
        }
        let name = names.lines().map_while(Result::ok).nth(idx);
        if name.is_none() {
            bail!("Name for system ID {id} (#{id64}) not found!")
        }
        Ok((id, name))
    }
}

// TODO: merge [router::LineCache], [data_loader::DataLoader] and
// [CompressedIndex], add Index field to [router::Router]
// TODO: compress procgen names based on prefix, suffix, infix list
// [sector_names.rs]

#[derive(Debug)]
enum ChunkType {
    Single(u32),
    Roaring(RoaringBitmap),
    BTreeSet(BTreeSet<u32>),
    DeltaU8(u32, Vec<u8>),
    DeltaU16(u32, Vec<u16>),
}

impl ChunkType {
    fn encode_inner(data: &[u32]) -> Self {
        if data.len() == 1 {
            return Self::Single(data[0]);
        }
        let max_diff = data
            .iter()
            .tuple_windows()
            .map(|(a, b)| b - a)
            .max()
            .unwrap_or_default();
        if max_diff <= 0xff {
            return Self::encode_delta_u8(data);
        }
        if max_diff <= 0xffff {
            return Self::encode_delta_u16(data);
        }
        return Self::BTreeSet(data.iter().copied().collect());
    }

    fn read(reader: &mut impl Read) -> Result<Self> {
        let id = reader.read_u8()?;
        match id {
            0 => Ok(Self::Single(reader.read_u32::<LittleEndian>()?)),
            1 => Ok(Self::Roaring(RoaringBitmap::deserialize_from(reader)?)),
            2 => {
                let size = reader.read_u32::<LittleEndian>()? as usize;
                let mut entries = vec![0u32; size];
                reader.read_u32_into::<LittleEndian>(&mut entries)?;
                Ok(Self::BTreeSet(entries.into_iter().collect()))
            }
            3 => {
                let first = reader.read_u32::<LittleEndian>()?;
                let size = reader.read_u32::<LittleEndian>()? as usize;
                let mut entries = vec![0; size];
                reader.read_exact(&mut entries)?;
                Ok(Self::DeltaU8(first, entries.into_iter().collect()))
            }
            4 => {
                let first = reader.read_u32::<LittleEndian>()?;
                let size = reader.read_u32::<LittleEndian>()? as usize;
                let mut entries = vec![0; size];
                reader.read_u16_into::<LittleEndian>(&mut entries)?;
                Ok(Self::DeltaU16(first, entries.into_iter().collect()))
            }
            id => bail!("Invalid chunk type ID: {id}"),
        }
    }

    fn write(&self, writer: &mut impl Write) -> Result<()> {
        writer.write_u8(self.id())?;
        match self {
            Self::Single(id) => writer.write_u32::<LittleEndian>(*id)?,
            Self::Roaring(r) => r.serialize_into(writer)?,
            Self::BTreeSet(s) => {
                writer.write_u32::<LittleEndian>(s.len().try_into()?)?;
                for entry in s {
                    writer.write_u32::<LittleEndian>(*entry)?;
                }
            }
            Self::DeltaU8(init, diff) => {
                writer.write_u32::<LittleEndian>(*init)?;
                writer.write_u32::<LittleEndian>(diff.len().try_into()?)?;
                writer.write_all(diff)?;
            }
            Self::DeltaU16(init, diff) => {
                writer.write_u32::<LittleEndian>(*init)?;
                writer.write_u32::<LittleEndian>(diff.len().try_into()?)?;
                for entry in diff {
                    writer.write_u16::<LittleEndian>(*entry)?;
                }
            }
        };
        Ok(())
    }

    const fn id(&self) -> u8 {
        match self {
            Self::Single(_) => 0,
            Self::Roaring(_) => 1,
            Self::BTreeSet(_) => 2,
            Self::DeltaU8(_, _) => 3,
            Self::DeltaU16(_, _) => 4,
        }
    }

    fn encode(data: &[u32]) -> Self {
        let roaring: RoaringBitmap = data.iter().collect();
        let roaring_size = roaring.serialized_size();
        let other = Self::encode_inner(data);
        if other.size() < roaring_size {
            return other;
        }
        return Self::Roaring(roaring);
    }

    fn size(&self) -> usize {
        1 + match self {
            Self::Single(_) => 4,
            Self::Roaring(b) => b.serialized_size(),
            Self::BTreeSet(set) => 4 + set.len() * 4,
            Self::DeltaU8(_, list) => 4 + 4 + list.len(),
            Self::DeltaU16(_, list) => 4 + 4 + list.len() * 2,
        }
    }

    fn encode_delta_u8(data: &[u32]) -> Self {
        let delta = data
            .iter()
            .tuple_windows()
            .map(|(a, b)| {
                u8::try_from(b - a).unwrap_or_else(|_| unreachable!())
            })
            .collect::<Vec<u8>>();
        Self::DeltaU8(data[0], delta)
    }

    fn encode_delta_u16(data: &[u32]) -> Self {
        let delta = data
            .iter()
            .tuple_windows()
            .map(|(a, b)| {
                u16::try_from(b - a).unwrap_or_else(|_| unreachable!())
            })
            .collect::<Vec<u16>>();
        Self::DeltaU16(data[0], delta)
    }
}

fn read_index(mm: &Mmap) -> Result<()> {
    let mut uid = vec![0u8; 16];
    let mut rd = Cursor::new(mm);
    rd.read_exact(&mut uid)?;
    let num_chunks = rd.read_u32::<LittleEndian>()? as usize;
    let mut index: Vec<(u32, u32)> = vec![(0, 0); num_chunks];
    let t_start = Instant::now();
    for res in index.iter_mut() {
        // ID Hi
        res.0 = rd.read_u32::<LittleEndian>()?;
        // Chunk Pos
        res.1 = rd.read_u32::<LittleEndian>()?;
    }
    info!("Index header read in {}", t_start.elapsed().human_duration());
    let mut ids: BTreeMap<u8, u64> = BTreeMap::new();
    for (.., chunk_pos) in index {
        rd.set_position(chunk_pos as u64);
        let chunk = ChunkType::read(&mut rd)?;
        *ids.entry(chunk.id()).or_default() += 1u64;
    }
    println!("{ids:?}");
    Ok(())
}

fn trigram_test() -> Result<()> {
    use rayon::prelude::*;
    let names_file =
        r"C:\Users\Earthnuker\AppData\Local\astronav\data\stars.names";
    let mut tg_idx = BufWriter::new(File::create(
        r"C:\Users\Earthnuker\AppData\Local\astronav\data\stars.names.tgidx",
    )?);
    let mm_names = unsafe { memmap2::Mmap::map(&File::open(names_file)?)? };
    // let mut tg_map: PatriciaMap<RoaringBitmap> = PatriciaMap::new();
    // for (id, name) in mm_names.split(|&c| c == b'\n').skip(1).enumerate() {
    //     for key in name.iter().copied().tuple_windows::<(u8, u8, u8)>() {
    //         let key: [u8; 3] = key.into();
    //         if !tg_map.contains_key(key) {
    //             tg_map.insert(key, RoaringBitmap::default());
    //         }
    //         let res = tg_map.get_mut(key).unwrap_or_else(|| unreachable!());
    //         res.insert(id.try_into()?);
    //     }
    // }
    // bincode::serialize_into(&mut tg_idx, &tg_map)?;
    Ok(())
}

fn compress_chunk(data: &[u8]) -> Result<Vec<u8>> {
    Ok(zstd::encode_all(Cursor::new(data), 22)?)
}

fn decompress_chunk(data: &[u8]) -> Result<Vec<u8>> {
    Ok(zstd::decode_all(Cursor::new(data))?)
}

#[allow(clippy::expect_used)]
fn names_test(regenrate: bool) -> Result<()> {
    use crate::data_loader::SearchResult;
    let needles = [
        // b"Pallaeni".as_slice(),
        // b"Omega Sector VE-Q b5-15".as_slice(),
        // b"Pru Aescs NC-M d7-192".as_slice(),
        // b"Clooku EW-Y c3-197".as_slice(),
        // b"Boewnst KS-S c20-959".as_slice(),
        // b"Dryau Ausms KG-Y e3390".as_slice(),
        // b"Stuemeae KM-W c1-342".as_slice(),
        // b"Hypiae Phyloi LR-C d22".as_slice(),
        // b"Phroi Bluae QI-T e3-3454".as_slice(),
        // b"Bleethuae NI-B d674".as_slice(),
        // b"Smootoae QY-S d3-202".as_slice(),
        b"Bagle Ppnoint".as_slice(),
        // b"Beagle Point".as_slice(),
    ]
    .into_iter()
    .map(SearchResult::new)
    .collect_vec();

    use rayon::prelude::*;
    let block_size = 1024 * 1024 * 4; // 4MB
    let names_file =
        r"C:\Users\Earthnuker\AppData\Local\astronav\data\stars.names";
    let compressed_file =
        r"C:\Users\Earthnuker\AppData\Local\astronav\data\stars.names.comp";
    if regenrate {
        let mm_names = unsafe { memmap2::Mmap::map(&File::open(names_file)?)? };
        let chunks = mm_names.split(|&c| c == b'\n').skip(1);
        let mut accumulator = Vec::with_capacity(block_size * 2);
        let mut out_fh = BufWriter::new(File::create(compressed_file)?);
        let mut base_id = 0;
        for (curr_id, chunk) in chunks.enumerate() {
            accumulator.extend(chunk);
            accumulator.push(b'\n');
            if accumulator.len() > block_size {
                println!(
                    "[{base_id}] {}",
                    std::str::from_utf8(chunk).unwrap_or_default()
                );
                let data = compress_chunk(&accumulator)?;
                out_fh.write_u32::<LittleEndian>(base_id.try_into()?)?;
                out_fh.write_u32::<LittleEndian>(data.len().try_into()?)?;
                out_fh.write_all(&data)?;
                Vec::clear(&mut accumulator);
                base_id = curr_id + 1;
            }
        }
        drop(out_fh);
    }
    let t_start = Instant::now();
    let mut idx = vec![];
    let mm = unsafe { memmap2::Mmap::map(&File::open(compressed_file)?)? };
    let mut c = Cursor::new(&mm);
    let mm_len: u64 = mm.len().try_into()?;
    while c.position() != mm_len {
        let base_id = c.read_u32::<LittleEndian>()?;
        let size = c.read_u32::<LittleEndian>()?;
        let pos: usize = c.position().try_into()?;
        c.seek(SeekFrom::Current(size.into()))?;
        idx.push((base_id, pos, size));
    }
    println!(
        "Compressed index for {} Chunks loaded in {}",
        idx.len(),
        t_start.elapsed().human_duration(),
    );
    let t_start = Instant::now();
    let mut acc = needles.clone();
    idx.into_par_iter()
        .fold_with(needles, |mut needles, (base_id, pos, size)| {
            let sys_id: usize = base_id.try_into().expect("Overflow in ID");
            let buf = decompress_chunk(&mm[pos..pos + size as usize])
                .unwrap_or_default();
            for (offset, entry) in buf.split(|&c| c == b'\n').enumerate() {
                for res in needles.iter_mut() {
                    res.update(
                        (sys_id + offset).try_into().expect("Overflow in ID"),
                        entry,
                    );
                }
            }
            needles
        })
        .collect::<Vec<_>>()
        .into_iter()
        .for_each(|results| {
            for (acc, res) in acc.iter_mut().zip(results.iter()) {
                acc.merge(res);
            }
        });
    dbg!(acc);

    println!(
        "Compressed index searched in {}",
        t_start.elapsed().human_duration(),
    );
    Ok(())
}

fn index_test() -> Result<()> {
    let old_index_file =
        r"C:\Users\Earthnuker\AppData\Local\astronav\data\stars.idx";
    let new_index_file =
        r"C:\Users\Earthnuker\AppData\Local\astronav\data\stars.cidx";
    let mut wr = BufWriter::new(File::create(new_index_file)?);
    let mm = unsafe { Mmap::map(&File::open(old_index_file)?) }?;
    let num_systems = (mm.len() - 16) / (8 + 4);
    let data: &[[u8; 8 + 4]] = unsafe {
        std::slice::from_raw_parts(mm[16..].as_ptr() as *const _, num_systems)
    };
    let mut prev_hi = u32::MAX;
    let mut prev_id64 = 0;
    let mut chunks: BTreeMap<u32, ChunkType> = BTreeMap::new();
    let mut current_chunk = vec![];
    let t_start = Instant::now();
    for rec in data {
        let id64 = parse_record(rec)?;
        assert!(
            id64 > prev_id64,
            "ID64s have to be sorted in ascending order!"
        );
        let (hi, lo) = split_id64(id64);
        current_chunk.push(lo);
        if prev_hi != hi {
            chunks.insert(hi, ChunkType::encode(&current_chunk));
            Vec::clear(&mut current_chunk);
        }
        prev_hi = hi;
        prev_id64 = id64;
    }
    wr.write_all(&mm[..16])?;
    wr.write_u32::<LittleEndian>(chunks.len().try_into()?)?;
    // UID + index_len + index_len * (id_hi + offset)
    let mut offset: u32 = (16 + 4 + (4 * 2 * chunks.len())).try_into()?;
    info!("Writing index for {n} chunks", n = chunks.len());
    for (id_hi, chunk) in chunks.iter() {
        wr.write_u32::<LittleEndian>(*id_hi)?;
        wr.write_u32::<LittleEndian>(offset)?;
        offset += u32::try_from(chunk.size())?;
    }
    info!("Writing {n} chunks", n = chunks.len());
    for chunk in chunks.into_values() {
        chunk.write(&mut wr)?;
    }
    info!("Done in {}", t_start.elapsed().human_duration());
    drop(wr);
    drop(mm);
    let mm = unsafe { Mmap::map(&File::open(new_index_file)?) }?;
    read_index(&mm)?;
    // let idx = CompressedIndex::load(&new_index_file)?;
    // let t_start = Instant::now();
    // dbg!(idx.get_by_id64(10477373803, true)?); // Sol
    // dbg!(idx.get_by_id64(688319, true)?); // Cygni X-3
    // dbg!(idx.get_by_id64(20578934, true)?); // Sagittarius A*
    // dbg!(idx.get_by_id64(19919648686512640, true)?); // Myriesly IL-R
    // a59-1132 // dbg!(idx.get_id64_by_id(178387)?);
    // info!("Query took {}", t_start.elapsed().human_duration());
    Ok(())
}
