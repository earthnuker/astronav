//! Spansh galaxy.json converter
use std::{
    borrow::Cow,
    convert::TryInto,
    fmt::Write as _,
    io::{BufRead, BufReader, BufWriter, Cursor, Read, Seek, Write},
    path::{Path, PathBuf},
    thread::JoinHandle,
    time::Instant,
};

use byteorder::{LittleEndian, WriteBytesExt};
use color_eyre::eyre::{anyhow, ensure, Context, Result};
use crossbeam_channel::{bounded, Receiver, SendError};
use flate2::read::GzDecoder;
use fs_err::File;
use human_repr::{HumanCount, HumanDuration};
use itertools::Itertools;
use lazy_static::lazy_static;
use rand::Rng;
use rustc_hash::FxHashSet;
use serde::Deserialize;
use sha3::Digest;
use tracing::{error, info, warn};

use crate::{
    common::{StarKind, System, SystemFlags, TreeNode},
    event::{Event, ProcessState},
    route::Router,
};

lazy_static! {
    static ref SCOOPABLE_TYPES: FxHashSet<char> = "KGBFOAM".chars().collect();
}

// TODO: implement updating based on the fact that entries are sorted by ID64
// TODO: Build KD-Tree in advance, skip generating ids in [data_loader::load]
// use [slice::elem_offset] to get ID

// Last-Modified header
// 1day
// 7days
// 1month // 32 days

static GALAXY_DUMP_URL: &str = "https://downloads.spansh.co.uk/galaxy.json.gz";

#[derive(Debug, Deserialize, Clone)]
struct GalaxyCoords {
    x: f32,
    y: f32,
    z: f32,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct GalaxyBody<'a> {
    #[serde(borrow, rename = "type")]
    body_type: &'a str,
    #[serde(borrow)]
    sub_type: Option<&'a str>,
    main_star: Option<bool>,
}

#[derive(Debug, Deserialize, Clone)]
struct GalaxySystem<'a> {
    id64: u64,
    coords: GalaxyCoords,
    #[serde(borrow)]
    name: Cow<'a, str>,
    bodies: Vec<GalaxyBody<'a>>,
}

fn get_sys_flags(sys: &GalaxySystem<'_>) -> SystemFlags {
    let mut primary_kind = StarKind::Regular;
    let mut kind = StarKind::Regular;
    for b in sys.bodies.iter().filter(|b| b.body_type == "Star") {
        let main_star = b.main_star.unwrap_or_default();
        let sub_type = b.sub_type.as_ref().unwrap_or(&"?");
        let is_scoopable = SCOOPABLE_TYPES
            .contains(&sub_type.chars().next().unwrap_or_default());
        let star_type = if is_scoopable {
            StarKind::Scoopable
        } else if sub_type.contains("Neutron") {
            StarKind::Neutron
        } else if sub_type.contains("White Dwarf") {
            StarKind::WhiteDwarf
        } else {
            StarKind::Regular
        };
        if main_star {
            primary_kind = primary_kind.max(star_type);
        } else {
            kind = kind.max(star_type);
        };
    }
    SystemFlags::new(primary_kind, kind)
}

fn parse_record(buffer: &str) -> Result<GalaxySystem<'_>> {
    Ok(serde_json::from_str::<GalaxySystem<'_>>(
        buffer.trim_end_matches(|c: char| c == ',' || c.is_whitespace()),
    )?)
}

fn convert_record(sys: &GalaxySystem<'_>, id: u32) -> System {
    System {
        id,
        id64: sys.id64,
        name: sys.name.to_string(),
        pos: [sys.coords.x, sys.coords.y, sys.coords.z],
        flags: get_sys_flags(sys),
    }
}

struct ReaderWrapper {
    reader: Box<dyn Read + Send + Sync>,
    pos: usize,
}

impl ReaderWrapper {
    fn new(reader: Box<dyn Read + Send + Sync>) -> Self {
        Self { reader, pos: 0 }
    }

    const fn get_pos(&self) -> usize {
        self.pos
    }
}

impl Read for ReaderWrapper {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let size = self.reader.read(buf)?;
        self.pos += size;
        Ok(size)
    }
}

struct GalaxyLoader {
    rx: Receiver<(u64, f64, String)>,
    total_len: f64,
    handle: JoinHandle<Result<()>>,
}

impl GalaxyLoader {
    fn new(path: &Path) -> Result<Self> {
        let fh = File::open(path)?;
        let total_len = fh.metadata()?.len() as f64;
        Ok(Self::build(Box::new(fh), total_len))
    }

    fn new_spansh() -> Result<Self> {
        let resp = ureq::get(GALAXY_DUMP_URL).call()?;
        let total_len: u64 = resp
            .header("Content-Length")
            .ok_or_else(|| anyhow!("Failed to get Content-Length header"))?
            .parse::<u64>()?;
        Ok(Self::build(resp.into_reader(), total_len as f64))
    }

    fn build<R: Read + Send + Sync + 'static>(fh: R, total_len: f64) -> Self {
        let (tx, rx) = bounded(1024);
        let mut gz_reader = BufReader::with_capacity(
            1024 * 1024, // 1MB
            GzDecoder::new(ReaderWrapper::new(Box::new(fh))),
        );
        let handle = std::thread::spawn(move || -> Result<()> {
            let mut bytes_read = 0u64;
            let mut buffer = String::with_capacity(1024*8);
            let mut records = 0u64;
            loop {
                let n = gz_reader.read_line(&mut buffer)?;
                if n == 0 {
                    return Ok(());
                }
                bytes_read += n as u64;
                let cur_pos = gz_reader.get_ref().get_ref().get_pos() as f64;
                
                if let Err(err) = tx.send((bytes_read, cur_pos, std::mem::take(&mut buffer))) {
                    error!("Failed to send record: {err}");
                    return Ok(());
                }
                records += 1;
                // if records>1_000_000 {
                //     return Ok(());
                // }
            }
        });
        Self { rx, total_len, handle }
    }

    fn close(self) -> Result<()> {
        drop(self.rx);
        self.handle
            .join()
            .map_err(|err| anyhow!("Failed to join thread: {:?}", err))?
    }
}

#[allow(clippy::copy_iterator)]
impl Iterator for &GalaxyLoader {
    type Item = (u64, f64, String);

    fn next(&mut self) -> Option<Self::Item> {
        self.rx.recv().ok()
    }
}

/// Load compressed galaxy.json from `path` and write `{bin,idx,names}` files to
/// `out_path`
pub fn process_galaxy_dump(
    path: Option<&PathBuf>,
    out_path: &Path,
    router: &Router,
) -> Result<()> {
    use crate::common::is_older;
    let mut prev_id64 = 0u64;
    let out_dat = out_path.with_extension("dat");
    let out_names = out_path.with_extension("names.tmp");
    let mut regenerate = true;
    if let Some(path) = path {
        for &out_path in &[&out_dat, &out_names] {
            if is_older(out_path, path) {
                info!(
                    "{} is older than {}, not regenerating",
                    path.display(),
                    out_path.display()
                );
                regenerate = false;
            }
        }
        if !regenerate {
            return Ok(());
        }
    }

    let mut state = ProcessState {
        prc_done: 0.0,
        bytes_rate: 0.0,
        sys_rate: 0.0,
        t_rem: 0.0,
        file_pos: 0,
        num_systems: 0,
        num_errors: 0,
        msg: None,
        index_size: 0,
        data_size: 0,
        names_size: 0,
        bytes_read: 0,
        uncomp_rate: 0.0,
    };
    let mut t_last = Instant::now()
        .checked_sub(router.status_interval)
        .unwrap_or_else(|| unreachable!());
    let mut data_wtr = BufWriter::new(File::create(&out_dat)?);
    let mut names_wtr = BufWriter::new(File::create(&out_names)?);
    let uid = {
        let mut uid = [0u8; 16];
        rand::thread_rng().fill(&mut uid[..]);
        data_wtr.write_all(&uid)?;
        let uid_str: String = uid.iter().map(|v| format!("{v:02X}")).collect();
        writeln!(&mut names_wtr, "#{uid_str}")?;
        uid
    };
    let mut offset = uid.len() * 2 + 2;
    let mut systems: u64 = 0;
    let t_start = Instant::now();
    let loader = if let Some(path) = path {
        GalaxyLoader::new(path)?
    } else {
        GalaxyLoader::new_spansh()?
    };
    for (bytes_read, cur_pos, buffer) in &loader {
        let sys = parse_record(&buffer).and_then(|sys| {
            Ok(convert_record(
                &sys,
                (systems + 1).try_into().context("System ID overflow")?,
            ))
        });
        match sys {
            Ok(sys) => {
                systems += 1;
                if t_last.elapsed() > router.status_interval {
                    t_last = Instant::now();
                    state.file_pos = cur_pos as u64;
                    state.bytes_read = bytes_read;
                    state.uncomp_rate =
                        (bytes_read as f64) / t_start.elapsed().as_secs_f64();
                    state.prc_done = (cur_pos / loader.total_len) * 100.0;
                    state.bytes_rate =
                        cur_pos / t_start.elapsed().as_secs_f64();
                    state.sys_rate =
                        (systems as f64) / t_start.elapsed().as_secs_f64();
                    state.t_rem =
                        (loader.total_len - cur_pos) / state.bytes_rate;
                    state.num_systems = systems;
                    state.data_size = data_wtr.get_ref().stream_position()?;
                    state.names_size = names_wtr.get_ref().stream_position()?;
                    if let Some(ref cb) = router.callback {
                        cb(router, &Event::ProcessState(state.clone()))?;
                    };
                }
                if sys.id64 <= prev_id64 {
                    warn!(
                        "Non-increasing ID64 found: {id64}<={prev_id64}",
                        id64 = sys.id64
                    );
                }
                data_wtr.write_u8(sys.flags.value())?;
                data_wtr.write_f32::<LittleEndian>(sys.pos[0])?;
                data_wtr.write_f32::<LittleEndian>(sys.pos[1])?;
                data_wtr.write_f32::<LittleEndian>(sys.pos[2])?;
                data_wtr.write_u64::<LittleEndian>(sys.id64)?;
                data_wtr.write_u32::<LittleEndian>(offset as u32)?;
                let name_bytes = sys.name.as_bytes();
                names_wtr.write_all(name_bytes)?;
                names_wtr.write_u8(b'\n')?;
                offset += name_bytes.len() + 1;
                prev_id64 = sys.id64;
            }
            Err(e) => {
                error!("{}: {:?}", e, buffer);
                state.num_errors += 1;
            }
        }
    }
    loader.close()?;
    let bin_pos = data_wtr.into_inner()?.stream_position()?;
    let names_pos = names_wtr.into_inner()?.stream_position()?;
    info!(
        "Wrote {} data and {} system names in {}",
        bin_pos.human_count_bytes(),
        names_pos.human_count_bytes(),
        t_start.elapsed().human_duration()
    );
    let t_start = Instant::now();
    make_tree(out_path, &uid)?;
    info!("Data reformattted in {}", t_start.elapsed().human_duration());
    Ok(())
}
/*
12598.99 -> 2m
309255.24 -> 10m
6397047.11 -> 1h
*/

#[repr(C, packed)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Node {
    pub flags: SystemFlags,
    pub pos: [f32; 3],
    pub id64: u64,
    pub name_offset: u32,
}

impl sif_kdtree::Object for Node {
    type Point = [f32; 3];

    fn position(&self) -> &[f32; 3] {
        unsafe { &*(&raw const self.pos).cast::<[f32; 3]>() }
    }
}

impl sif_kdtree::Object for &Node {
    type Point = [f32; 3];

    fn position(&self) -> &[f32; 3] {
        unsafe { &*(&raw const self.pos).cast::<[f32; 3]>() }
    }
}

pub(crate) fn make_tree<P: AsRef<Path>>(
    out_path: P,
    uid: &[u8; 16],
) -> Result<()> {
    use fs_err::OpenOptions;
    use memmap2::{Mmap, MmapOptions};
    use rayon::prelude::*;

    use crate::data_loader::HEADER_SIZE;
    let out_path = out_path.as_ref();
    let in_path = out_path.with_extension("dat");
    let names_path = out_path.with_extension("names.tmp");
    let names = File::open(&names_path)?;
    let mm_names = unsafe { Mmap::map(&names)? };
    let mut fh_names = Cursor::new(mm_names);
    drop(names);
    let fh = OpenOptions::new()
        .read(true)
        .write(true)
        .truncate(false)
        .open(&in_path)?;
    let mut mm = unsafe {
        MmapOptions::new()
            .offset(HEADER_SIZE.try_into()?)
            .map(&fh)?
            .make_mut()?
    };
    drop(fh);
    let nodes: &mut [Node] = bytemuck::cast_slice_mut(&mut mm);
    let t_start = Instant::now();
    sif_kdtree::KdTree::par_new(&mut *nodes);
    info!(
        "Built KD-Tree in {}",
        t_start.elapsed().human_duration()
    );
    let out_bin = out_path.with_extension("bin");
    let out_names = out_path.with_extension("names");
    let out_index = out_path.with_extension("idx");
    let mut data_wtr = BufWriter::new(File::create(&out_bin)?);
    data_wtr.write_all(uid)?;
    let mut index_wtr = BufWriter::new(File::create(&out_index)?);
    index_wtr.write_all(uid)?;
    let mut names_wtr = BufWriter::new(File::create(&out_names)?);
    let uid_str: String = uid.iter().map(|v| format!("{v:02X}")).collect();
    writeln!(&mut names_wtr, "#{uid_str}")?;
    let mut name_buf = String::new();
    let mut names_offset: u32 = u32::try_from(uid.len())?*2+2;
    let mut needs_nl: bool;
    for node in nodes.iter() {
        fh_names.seek(std::io::SeekFrom::Start(node.name_offset as u64))?;
        fh_names.read_line(&mut name_buf)?;
        names_wtr.write_all(name_buf.as_bytes())?;
        needs_nl=name_buf.bytes().last() != Some(b'\n');
        if needs_nl {
            names_wtr.write_u8(b'\n')?;
        }
        data_wtr.write_u8(node.flags.value())?;
        data_wtr.write_f32::<LittleEndian>(node.pos[0])?;
        data_wtr.write_f32::<LittleEndian>(node.pos[1])?;
        data_wtr.write_f32::<LittleEndian>(node.pos[2])?;
        index_wtr.write_u64::<LittleEndian>(node.id64)?;
        index_wtr.write_u32::<LittleEndian>(names_offset)?;
        names_offset+=name_buf.len() as u32 + (needs_nl as u32);
        name_buf.clear();
    }
    drop(mm);
    drop(fh_names);
    std::fs::remove_file(in_path)?;
    std::fs::remove_file(names_path)?;
    Ok(())
}
