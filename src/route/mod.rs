#![allow(clippy::unwrap_in_result)]
//! Route computation functions using various graph search algorithms
use core::panic;
use std::{
    collections::{BTreeMap, VecDeque},
    hash::{Hash, Hasher},
    io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    marker::PhantomData,
    ops::ControlFlow,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
    u32, usize,
};

use byteorder::{LittleEndian, ReadBytesExt};
use color_eyre::eyre::{self, bail, Context, OptionExt as _, Result};
use fs_err::File;
use human_repr::{HumanCount, HumanDuration, HumanThroughput};
use itertools::Itertools;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use sif_kdtree::WithinDistance;
use tracing::*;

use crate::{
    common::{
        dist, dist2, AstronavError, AstronavResult, BeamWidth, FormatFloat,
        FormatNum, SysEntry, System, SystemFlags, TreeNode, F32, ID64,
    },
    data_loader,
    data_loader::MappedNodes,
    event::{Callback, Event, RouteState},
    ship::NamedShip,
};

mod mode;
pub(crate) use mode::*;

mod algorithms;

type TreeDiff = (u32, u32, Vec<(u32, u32)>);

pub(crate) type KdTree<T = TreeNode, O = NodeStore> = sif_kdtree::KdTree<T, O>;

#[derive(Deserialize, Serialize)]
enum PrecompTree {
    Full { id: u32, map: Vec<u32> },
    Partial { parent: u32, diff: Vec<(u32, u32)> },
}

// struct Weight {
//     dist_from_start: f32,
//     dist_to_goal: f32,
//     dist_to_point: Vec<(f32, [f32; 3])>,
// }

// impl Weight {
//     fn calc(&self, node: &TreeNode, dst: &TreeNode, src: &TreeNode) -> f32 {
//         let d_total = dist(&src.pos, &dst.pos);
//         let d_start =
//             (dist(&node.pos, &src.pos) / d_total) * self.dist_from_start;
//         let d_goal = (dist(&node.pos, &dst.pos) / d_total) *
// self.dist_to_goal;         let points: f32 = self
//             .dist_to_point
//             .iter()
//             .map(|&(f, p)| dist(&p, &node.pos) * f)
//             .sum();
//         d_start + d_goal + points
//     }
// }

const fn default_weight() -> F32 {
    F32(1.0)
}

#[derive(Debug)]
pub enum PrecomputeMode {
    // Full,
    // RouteFrom(u32),
    // RouteTo(u32),
}

impl TreeNode {
    pub fn dist2(&self, p: &[f32; 3]) -> f32 {
        dist2(self.pos(), p)
    }

    pub fn distp(&self, p: &Self) -> f32 {
        dist(self.pos(), p.pos())
    }
}

impl PartialEq for System {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for System {}

impl Hash for System {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

fn hash_file(path: &Path) -> Result<Vec<u8>> {
    let mut hash_reader = BufReader::new(File::open(path)?);
    let mut hasher = Sha3_256::new();
    std::io::copy(&mut hash_reader, &mut hasher)?;
    Ok(hasher.finalize().iter().copied().collect())
}

// TODO: replace Files with Mmaps
#[derive(Debug)]
pub struct LineCache {
    data_reader: BufReader<File>,
    name_reader: BufReader<File>,
    index_reader: BufReader<File>,
    data_len: u64,
    index_len: u64,
}

impl LineCache {
    const SKIP: u64 = 16; // UID length
    const DATA_RECORD_SIZE: u64 = 1 + 4 * 3; // flags/num_bodies + pos_x, pos_y, pos_z
    const INDEX_RECORD_SIZE: u64 = 8 + 4; // id64 + u32 name offset
    pub fn new(path: &Path) -> AstronavResult<Arc<Mutex<Self>>> {
        let mut data_reader =
            BufReader::new(File::open(path.with_extension("bin"))?);
        let mut name_reader =
            BufReader::new(File::open(path.with_extension("names"))?);
        let mut index_reader =
            BufReader::new(File::open(path.with_extension("idx"))?);
        let data_len = data_reader.get_ref().metadata()?.len();
        let index_len = index_reader.get_ref().metadata()?.len();
        let mut data_uid = vec![0u8; 16];
        let mut index_uid = vec![0u8; 16];
        let mut name_uid = vec![0u8; 1 + 16 * 2]; // '#' + 16 bytes as hex
        data_reader.read_exact(&mut data_uid)?;
        index_reader.read_exact(&mut index_uid)?;
        name_reader.read_exact(&mut name_uid)?;
        name_uid = hex::decode(&name_uid[1..])
            .map_err(|e| AstronavError::Other(e.into()))?;
        if !(name_uid == index_uid && index_uid == data_uid) {
            warn!("name:  {}", hex::encode(name_uid));
            warn!("index: {}", hex::encode(index_uid));
            warn!("data:  {}", hex::encode(data_uid));
            return Err(AstronavError::Other(eyre::anyhow!(
                "UID mismatch in data files"
            )));
        }
        Ok(Arc::new(Mutex::new(Self {
            data_reader,
            name_reader,
            index_reader,
            data_len,
            index_len,
        })))
    }

    const fn offset(&self, id: u32) -> (u64, u64) {
        (
            Self::SKIP + (id as u64) * Self::DATA_RECORD_SIZE,
            Self::SKIP + (id as u64) * Self::INDEX_RECORD_SIZE,
        )
    }

    pub fn get(&mut self, id: u32) -> AstronavResult<System> {
        let mut sys = System {
            id,
            id64: 0x5AFEC0DE5AFEC0DE,
            name: String::new(),
            flags: Default::default(),
            pos: [std::f32::NAN; 3],
        };
        let (data_pos, index_pos) = self.offset(id);
        if data_pos > self.data_len || index_pos > self.index_len {
            return Err(AstronavError::SystemNotFoundError(SysEntry::ID(id)));
        }
        self.data_reader.seek(SeekFrom::Start(data_pos))?;
        sys.flags = SystemFlags::from_value(self.data_reader.read_u8()?);
        self.data_reader.read_f32_into::<LittleEndian>(&mut sys.pos)?;
        self.index_reader.seek(SeekFrom::Start(index_pos))?;
        sys.id64 = self.index_reader.read_u64::<LittleEndian>()?;
        let offset = self.index_reader.read_u32::<LittleEndian>()? as u64;
        self.name_reader.seek(SeekFrom::Start(offset))?;
        self.name_reader.read_line(&mut sys.name)?;
        sys.name = sys.name.trim_end_matches('\n').to_owned();
        Ok(sys)
    }
}

pub enum DataInterface {
    // TODO: Implement DataInterface combining CompressedIndex, LineCache and
    // DataLoader
}

// OLD
// 8192: Took: 54.09 s, 898 k systems/s, 14.3 M queries/s
//  inf: Took: 4:49.3, 280.3 k systems/s, 10.9 M queries/s

// NEW
// 8192: Took: 24.46 s, 2 M systems/s, 30.6 M queries/s
// inf: 133,

struct MMSlice<T> {
    mm: memmap2::Mmap,
    _t: PhantomData<T>,
}

impl<T> MMSlice<T> {
    fn new<P: AsRef<Path>, const HEADER_SIZE: u64>(
        path: P,
    ) -> Result<MMSlice<T>> {
        use memmap2::MmapOptions;
        let fh = File::open(path.as_ref())?;
        let mm = unsafe { MmapOptions::new().offset(HEADER_SIZE).map(&fh)? };
        Ok(Self { mm, _t: PhantomData })
    }
}

impl<T> AsRef<[T]> for MMSlice<T> {
    fn as_ref(&self) -> &[T] {
        assert_eq!(
            self.mm.len() % std::mem::size_of::<T>(),
            0,
            "Data not aligned!"
        );
        unsafe {
            std::slice::from_raw_parts(
                self.mm.as_ptr().cast::<T>(),
                self.mm.len() / std::mem::size_of::<T>(),
            )
        }
    }
}

pub(crate) enum NodeStore {
    MMap(MappedNodes),
    Slice(Box<[TreeNode]>),
}

impl From<Box<[TreeNode]>> for NodeStore {
    fn from(v: Box<[TreeNode]>) -> Self {
        Self::Slice(v)
    }
}

impl From<MappedNodes> for NodeStore {
    fn from(v: MappedNodes) -> Self {
        Self::MMap(v)
    }
}

impl Default for NodeStore {
    fn default() -> Self {
        Self::Slice(Default::default())
    }
}

impl AsRef<[TreeNode]> for NodeStore {
    fn as_ref(&self) -> &[TreeNode] {
        match self {
            Self::MMap(mm) => mm.as_ref(),
            Self::Slice(slice) => slice.as_ref(),
        }
    }
}

#[derive(Default)]
pub(crate) struct RouterTree(KdTree);

impl From<KdTree> for RouterTree {
    fn from(tree: KdTree) -> Self {
        Self(tree)
    }
}

impl std::ops::DerefMut for RouterTree {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl std::ops::Deref for RouterTree {
    type Target = KdTree<TreeNode>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl RouterTree {
    // TODO: Replace node.id with tree.id(&node)
    #[inline(always)]
    pub(crate) fn id(&self, node: &TreeNode) -> u32 {
        self
            .element_offset(node)
            .unwrap_or_else(|| panic!("Node {node:?} passed to RouterTree::id was not part of the KD-Tree"))
            .try_into()
            .unwrap_or(u32::MAX)
    }
    pub(crate) fn resolve(&self, sys: &System) -> Result<&TreeNode> {
        let node = self.nearest(&sys.pos).ok_or_eyre("Emptry tree!")?;
        let d = dist(node.pos(), &sys.pos);
        if d > 100.0 {
            let sys = sys.name.as_str();
            let d = d.human_count("Ly");
            warn!("Node for system {sys} is {d} away from choosen start system, this should not happen!");
        }
        Ok(node)
    }
}

pub struct Router {
    tree: Arc<RouterTree>,
    mode: ModeConfig,
    deterministic: bool,
    pub interrupted: AtomicBool,
    route_tree: Option<FxHashMap<u32, u32>>,
    cache: Option<Arc<Mutex<LineCache>>>,
    pub path: PathBuf,
    pub callback: Option<Callback>,
    hops: Vec<[f32; 3]>,
    dist: f32,
    max_id: usize,
    pub status_interval: Duration,
    ship: Option<NamedShip>,
    data: Option<DataInterface>,
}

impl std::fmt::Display for Router {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ship = self
            .ship
            .as_ref()
            .map_or_else(|| "<None>".to_string(), |ship| format!("{ship}"));
        write!(
            f,
            "Router(Stars: {stars}, Ship: {ship})",
            stars = self.tree.len()
        )?;
        Ok(())
    }
}

impl std::fmt::Debug for Router {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let tree =
            format!("<Tree with {} nodes>", self.tree.len().format_num());
        let callback = &self.callback.as_ref().map(|_| "<Callback>");
        let route_tree = &self.route_tree.as_ref().map(|tree| {
            format!("<RouteTree with {} edges>", tree.len().format_num())
        });
        f.debug_struct("Router")
            .field("tree", &tree)
            .field("mode", &self.mode)
            .field("route_tree", &route_tree)
            .field("cache", &self.cache)
            .field("path", &self.path)
            .field("callback", &callback)
            .field("hops", &self.hops)
            .field("dist", &self.dist)
            .field("total_nodes", &self.max_id.format_num())
            .field("status_interval", &self.status_interval)
            .field("ship", &self.ship)
            .finish()
    }
}

impl Default for Router {
    fn default() -> Self {
        let default_callback = Box::new(move |_: &Router, event: &Event| {
            info!("{event}");
            Ok(())
        });
        Self {
            interrupted: AtomicBool::new(false),
            tree: Arc::default(),
            route_tree: None,
            cache: None,
            callback: Some(default_callback),
            path: PathBuf::from(""),
            hops: vec![],
            max_id: 0,
            dist: 0.0,
            ship: None,
            mode: Default::default(),
            deterministic: true,
            status_interval: Duration::from_secs_f64(0.5),
            data: None,
        }
    }
}

impl Router {
    pub fn new(status_interval: Duration) -> Self {
        Self { status_interval, ..Default::default() }
    }

    pub fn set_callback(&mut self, callback: Callback) {
        self.callback = Some(callback);
    }

    pub fn set_deterministic(&mut self, flag: bool) {
        self.deterministic = flag;
    }

    pub fn unload(&mut self) {
        self.path = PathBuf::from("");
        self.tree = Arc::default();
    }

    pub fn set_path<P: AsRef<Path>>(&mut self, path: P) -> AstronavResult<()> {
        self.path = PathBuf::from(path.as_ref());
        self.cache = Some(LineCache::new(&self.path)?);
        Ok(())
    }

    pub fn set_hops(&mut self, hops: &[[f32; 3]]) {
        self.hops = hops.to_owned();
    }

    pub fn set_ship(&mut self, ship: NamedShip) {
        self.ship = Some(ship);
    }

    pub const fn get_ship(&self) -> Option<&NamedShip> {
        self.ship.as_ref()
    }

    pub fn load(
        &mut self,
        mmap_tree: bool,
        hops: &[[f32; 3]],
        dist: f32,
    ) -> AstronavResult<()> {
        if self.tree.len() == 0 {
            self.hops = hops.to_vec();
        }
        let hops_equal = hops.iter().all(|h| self.hops.contains(h))
            && self.hops.iter().all(|h| hops.contains(h))
            || dist == 0.0;
        if hops_equal && self.tree.len() != 0 {
            return Ok(());
        }
        let src = if mmap_tree {
            format!("Mapping [{}] into memory", self.path.display())
        } else {
            format!("Loading {} from disk", self.path.display())
        };
        if dist != 0.0 {
            info!(
                "{src}, max deviation from straight-line distance between hops: {}",
                dist.human_count(" Ly")
            );
        } else {
            info!("{src}");
        }
        let t_load = Instant::now();
        let (nodes, max_id) = if mmap_tree {
            let (nodes, max_id) = data_loader::map(&self.path)?;
            (NodeStore::from(nodes), max_id)
        } else {
            let (nodes, max_id) = data_loader::load(&self.path, hops, dist)?;
            (NodeStore::from(nodes), max_id)
        };
        // dbg!(systems.len(),max_id);
        self.max_id = max_id;
        let rate = (max_id as f64) / t_load.elapsed().as_secs_f64();
        info!(
            "{} Systems loaded in {:.2}, {:.2}",
            max_id.format_num(),
            t_load.elapsed().human_duration(),
            rate.human_throughput(" systems")
        );
        let t_build = Instant::now();
        self.tree = Arc::new(RouterTree::from(
            KdTree::<TreeNode>::new_unchecked(nodes),
        ));
        let rate = (max_id as f64) / t_load.elapsed().as_secs_f64();
        info!(
            "KD-Tree built in {:.2}, {:.2}",
            t_build.elapsed().human_duration(),
            rate.human_throughput(" systems")
        );
        self.cache = Some(LineCache::new(&self.path)?);
        info!("Total time: {:.2}", t_load.elapsed().human_duration());
        Ok(())
    }

    pub fn get(&self, id: u32) -> AstronavResult<System> {
        match self.cache.as_ref().map(|c| c.lock()) {
            Some(Ok(ref mut cache)) => cache.get(id),
            Some(Err(lock_err)) => Err(AstronavError::RuntimeError(format!(
                "Failed to lock cache: {lock_err}"
            ))),
            None => {
                Err(AstronavError::RuntimeError("No cache loaded".to_owned()))
            }
        }
    }

    pub fn get_node(&self, id: u32) -> AstronavResult<&TreeNode> {
        self.tree
            .get(id as usize)
            .ok_or_else(|| AstronavError::SystemNotFoundError(SysEntry::ID(id)))
    }

    pub fn get_tree(&self) -> Arc<RouterTree> {
        self.tree.clone()
    }

    pub fn from_file(filename: &Path) -> Result<(PathBuf, f32, Self)> {
        let mut reader = BufReader::new(match File::open(filename) {
            Ok(fh) => fh,
            Err(e) => {
                bail!("Error opening file {}: {}", filename.display(), e);
            }
        });
        info!("Loading {}", filename.display());
        let (range, file_hash, path, route_tree): (
            f32,
            Vec<u8>,
            PathBuf,
            FxHashMap<u32, u32>,
        ) = match bincode::deserialize_from(&mut reader) {
            Ok(res) => res,
            Err(e) => {
                bail!("Error loading file {}: {}", filename.display(), e);
            }
        };
        if hash_file(&path)? != file_hash {
            bail!("File hash mismatch!");
        }
        let cache =
            Some(LineCache::new(&path).context("Error creating cache")?);
        let total_nodes = route_tree.len();
        Ok((
            path.clone(),
            range,
            Self {
                interrupted: AtomicBool::new(false),
                tree: Arc::default(),
                route_tree: Some(route_tree),
                cache,
                path,
                callback: None,
                hops: vec![],
                max_id: total_nodes,
                dist: 0.0,
                ship: None,
                deterministic: true,
                status_interval: Duration::from_secs_f32(0.5),
                mode: Default::default(),
                data: None,
            },
        ))
    }

    pub fn closest(&self, center: &[f32; 3]) -> AstronavResult<System> {
        self.tree
            .nearest(center)
            .ok_or(AstronavError::SystemNotFoundError(SysEntry::Pos(
                center[0], center[1], center[2],
            )))?
            .get(self)
    }

    fn points_in_sphere<'a>(
        &'a self,
        center: [f32; 3],
        radius: f32,
        mut func: impl FnMut(&'a TreeNode),
    ) {
        // self.tree.locate_within_distance(*center, radius * radius)
        self.tree.look_up(
            &WithinDistance::new(center, radius),
            |o: &'a TreeNode| {
                (func)(o);
                ControlFlow::Continue(())
            },
        );
    }

    pub fn neighbours<'a>(
        &'a self,
        node: &'a TreeNode,
        range: f32,
        func: impl FnMut(&'a TreeNode),
    ) {
        self.points_in_sphere(*node.pos(), range * node.mult(), func);
    }

    fn initial_state(
        &self,
        src: &System,
        dest: &System,
        workers: usize,
    ) -> RouteState {
        RouteState {
            mode: self.mode,
            workers,
            from: src.name.to_owned(),
            to: dest.name.to_owned(),
            d_rem: dist(&src.pos, &dest.pos),
            d_total: dist(&src.pos, &dest.pos),
            system: String::default(),
            depth: 0,
            queue_size: 0,
            prc_done: 0.0,
            n_seen: 0,
            prc_seen: 0.0,
            rate: 0.0,
            refuels: None,
            msg: None,
        }
    }

    fn reconstruct(
        &self,
        goal_id: u32,
        map: &FxHashMap<u32, u32>,
        refuels: &FxHashSet<u32>,
    ) -> AstronavResult<Vec<System>> {
        let mut seen = Vec::new();
        let mut current = goal_id;
        seen.push(current);
        let mut path = vec![self.get(current)?];
        if refuels.contains(&path[0].id) {
            path[0].flags.set_refuel(true);
        }
        while let Some(&prev) = map.get(&current) {
            if seen.contains(&prev) {
                seen.push(prev);
                return Err(AstronavError::RuntimeError(format!(
                    "Found loop during path reconstruction: {seen:?}"
                )));
            }
            seen.push(prev);
            let mut sys = self.get(prev)?;
            if refuels.contains(&sys.id) {
                sys.flags.set_refuel(true);
            }
            path.push(sys);
            current = prev;
        }
        path.reverse();
        Ok(path)
    }

    pub fn resolve(
        &self,
        entries: &[SysEntry],
    ) -> AstronavResult<Vec<Option<System>>> {
        let mut entries = entries.to_vec();
        for ent in entries.iter_mut() {
            if let SysEntry::Name(name) = ent {
                if let Ok(id) = name.parse() {
                    *ent = id;
                }
            }
        }
        if !entries.iter().all(|e| matches!(e, SysEntry::ID(_))) {
            info!("Resolving {} systems", entries.len());
        }
        let mut names: Vec<String> = Vec::new();
        let mut id64s: Vec<u64> = Vec::new();
        let mut ret: Vec<Option<u32>> = vec![None; entries.len()];
        for ent in entries.iter_mut() {
            match &ent {
                SysEntry::ID64(id64) => id64s.push(*id64),
                SysEntry::Name(name) => names.push(name.to_owned()),
                _ => {}
            }
        }
        let name_ids = if !names.is_empty() {
            let t_start = Instant::now();
            let names_dedup = {
                let mut names = names.clone();
                names.sort();
                names.dedup();
                names
            };
            let (rate, res) = data_loader::search(&self.path, &names_dedup)?;
            info!(
                "Resolving names took: {:.2}, {:.2}",
                t_start.elapsed().human_duration(),
                rate.human_throughput(" systems")
            );
            res
        } else {
            FxHashMap::default()
        };
        let mut id64_ids = if !id64s.is_empty() {
            let t_start = Instant::now();
            let (rate, res) = data_loader::get_by_id64(&self.path, &id64s)?;
            info!(
                "Resolving ID64s took: {:.2}, {:.2}",
                t_start.elapsed().human_duration(),
                rate.human_throughput(" systems")
            );
            res
        } else {
            FxHashMap::default()
        };

        let unresolved_ids: Vec<u64> = id64s
            .iter()
            .filter(|v| !id64_ids.contains_key(v))
            .copied()
            .collect();

        if !unresolved_ids.is_empty() {
            warn!(
                "Failed to resolve some ID64s: {unresolved_ids:?}, getting closest star systems from embedded position data..."
            );
            for id64 in unresolved_ids.into_iter() {
                match ID64::try_from(id64) {
                    Ok(id) => {
                        let pos = id.coords();
                        let sys = self.tree.nearest(&pos).ok_or_else(|| {
                            AstronavError::SystemNotFoundError(SysEntry::Pos(
                                pos[0], pos[1], pos[2],
                            ))
                        })?;
                        let id = self.id(sys)?;
                        id64_ids.insert(id64, id);
                    }
                    Err(_) => {
                        unreachable!()
                    }
                }
            }
        }
        for (ent, ret_ent) in entries.iter().zip(ret.iter_mut()) {
            match ent {
                SysEntry::Name(name) => {
                    *ret_ent =
                        name_ids.get(name).map(|res| res.best_index).flatten()
                }
                SysEntry::ID64(id64) => *ret_ent = id64_ids.get(id64).copied(),
                SysEntry::ID(id) => *ret_ent = Some(*id),
                SysEntry::Pos(x, y, z) => {
                    let sys =
                        self.tree.nearest(&[*x, *y, *z]).ok_or_else(|| {
                            AstronavError::SystemNotFoundError(SysEntry::Pos(
                                *x, *y, *z,
                            ))
                        })?;
                    *ret_ent = Some(self.id(sys)?);
                }
            }
        }
        let mut lc = self
            .cache
            .as_ref()
            .ok_or_else(|| {
                AstronavError::RuntimeError("No LineCache available".to_owned())
            })?
            .lock()
            .map_err(|e| AstronavError::RuntimeError(e.to_string()))?;
        Ok(ret
            .into_iter()
            .map(|id| id.and_then(|id| lc.get(id).ok()))
            .collect_vec())
    }

    fn multiroute(
        &mut self,
        waypoints: &[System],
        range: f32,
        mode: ModeConfig,
    ) -> AstronavResult<Vec<System>> {
        if self.tree.len() == 0 {
            return Err(AstronavError::RuntimeError(
                "No Systems loaded, pleased load some with the 'load' method!"
                    .to_string(),
            ));
        }
        self.mode = mode;
        let mut route = vec![];
        if waypoints.len() < 2 {
            return Ok(waypoints.to_vec());
        }
        let total_legs = waypoints.len() - 1;
        for (mut current_leg,(src,dst)) in waypoints.iter().tuple_windows().enumerate() {
            current_leg+=1;
            if self.is_interrupted() {
                break;
            }
            let d_total = dist(&src.pos, &dst.pos);
            info!(
                "Plotting route from [{}] to [{}] (Leg {current_leg}/{total_legs})...",
                src.name, dst.name
            );
            debug!("Mode: {:?}", mode);
            info!("Mode: {}", mode);
            match mode {
                ModeConfig::BeamSearch { beam_width: BeamWidth::Absolute(n), ..} if n < 64 => {
                    warn!("Beam-width is less than 64, route computation might fail!");
                }
                _ => {}
            }
            if let Some(ship) = self.ship.as_ref() {
                info!("Using Ship: {ship}");
            }
            info!(
                "Jump range: {} Ly, distance: {} Ly, estimated jumps: {}",
                range.format_float(),
                d_total.format_float(),
                (d_total / range).format_float()
            );
            let t_leg = Instant::now();
            let leg = match &mode {
                ModeConfig::IncrementalBeamSearch { beam_width } => {
                    self.route_ibs(src, dst, range, *beam_width)
                }
                ModeConfig::AStar { weight } => {
                    self.route_astar(src, dst, range, **weight)
                }
                ModeConfig::DepthFirst => {
                    self.route_dfs(src, dst, range)
                }
                ModeConfig::BeamSearch {
                    beam_width,
                    refuel_mode,
                    refuel_primary,
                    boost_primary,
                    range_limit,
                } => self.route_beam(
                    src,
                    dst,
                    range,
                    beam_width,
                    refuel_mode.as_ref(),
                    *refuel_primary,
                    *boost_primary,
                    *range_limit,
                ),
                ModeConfig::BeamStack => {
                    self.route_beam_stack(src, dst, range)
                }
                ModeConfig::IncrementalBroadening => {
                    self.route_incremental_broadening(src, dst, range)
                }
                ModeConfig::Dijkstra => {
                    self.route_dijkstra(src, dst, range)
                }
                ModeConfig::Ship { ship_mode } => {
                    self.route_ship(src, dst, ship_mode)
                }
            }
            .map_err(|err| {
                AstronavError::RouteError {
                    from: src.clone(),
                    to: dst.clone(),
                    reason: Box::new(AstronavError::Other(err)),
                }
            })?;
            let leg_distance: f32 = leg
                .windows(2)
                .map(|w| dist(&w[0].pos, &w[1].pos))
                .sum();
            info!(
                "Leg {current_leg}/{total_legs} completed in {}: {} jumps, {} Ly",
                t_leg.elapsed().human_duration(),
                leg.len(),
                leg_distance.format_float()
            );
            if route.is_empty() {
                for sys in leg.iter() {
                    route.push(sys.clone());
                }
            } else {
                for sys in leg.iter().skip(1) {
                    route.push(sys.clone());
                }
            }
        }
        let waypoints =
            waypoints.iter().map(|w| w.id).collect::<FxHashSet<u32>>();
        for node in route.iter_mut() {
            node.flags.set_waypoint(waypoints.contains(&node.id));
        }
        Ok(route)
    }

    pub fn compute_best_diff(
        &self,
        paths: &[&str],
    ) -> AstronavResult<TreeDiff> {
        // let inverse_spt = FxHashMap<u32, FxHashSet<u32>>
        let mut trees = Vec::new();
        for &path in paths {
            let reader = BufReader::new(File::open(path)?);
            let spt: PrecompTree = bincode::deserialize_from(reader)?;
            let spt = match spt {
                PrecompTree::Full { id, map } => (id, map),
                PrecompTree::Partial { .. } => {
                    return Err("Need full tree!".to_owned().into());
                }
            };
            trees.push(spt);
        }
        let mut best = (std::usize::MAX, (0, 0, vec![]));
        for (i1, (id_1, t1)) in trees.iter().enumerate() {
            for (_i2, (id_2, t2)) in trees.iter().enumerate().skip(i1 + 1) {
                if t1.len() != t2.len() {
                    println!("Length missmatch between {id_1} and {id_2}");
                    continue;
                }
                let diff: Vec<(u32, u32)> = t1
                    .iter()
                    .zip(t2)
                    .enumerate()
                    .filter(|(_, (a, b))| a != b)
                    .map(|(i, (_, b))| (i as u32, *b))
                    .collect();
                if diff.len() < best.0 {
                    best = (diff.len(), (*id_1, *id_2, diff));
                }
            }
        }
        Ok(best.1)
    }

    fn get_systems_by_ids(
        &self,
        ids: &[u32],
    ) -> AstronavResult<FxHashMap<u32, System>> {
        let mut ret = FxHashMap::default();
        let mut c = self
            .cache
            .as_ref()
            .ok_or_else(|| {
                AstronavError::RuntimeError("No cache opened!".into())
            })?
            .lock()
            .map_err(|e| format!("Failed to lock cache: {e}"))?;
        for &id in ids {
            ret.insert(id, c.get(id)?);
        }
        Ok(ret)
    }

    fn route_to(&self, dst: &System) -> AstronavResult<Vec<System>> {
        if self.route_tree.is_none() {
            return Err(AstronavError::RuntimeError(
                "Can't compute route without a precomputed route-tree"
                    .to_owned(),
            ));
        }
        let prev = self
            .route_tree
            .as_ref()
            .ok_or_else(|| "No routing tree loaded".to_owned())?;
        if !prev.contains_key(&dst.id) {
            return Err(AstronavError::SystemNotFoundError(SysEntry::ID(
                dst.id,
            )));
        };
        let mut v_ids: Vec<u32> = Vec::new();
        let mut v: Vec<System> = Vec::new();
        let mut curr_sys: u32 = dst.id;
        loop {
            v_ids.push(curr_sys);
            match prev.get(&curr_sys) {
                Some(sys_id) => curr_sys = *sys_id,
                None => {
                    break;
                }
            }
        }
        v_ids.reverse();
        let id_map = self.get_systems_by_ids(&v_ids)?;
        for sys_id in v_ids {
            let sys = match id_map.get(&sys_id) {
                Some(sys) => sys,
                None => {
                    return Err(AstronavError::SystemNotFoundError(
                        SysEntry::ID(sys_id),
                    ));
                }
            };
            v.push(sys.clone())
        }
        Ok(v)
    }

    pub(crate) fn interrupt(&self) {
        self.interrupted.store(true, Ordering::SeqCst);
    }

    fn is_interrupted(&self) -> bool {
        self.interrupted.load(Ordering::SeqCst)
    }

    pub(crate) fn emit(&self, event: &Event) -> Result<bool> {
        if let Some(cb) = &self.callback {
            cb(self, event)?;
        }
        Ok(self.is_interrupted())
    }

    pub fn compute_route(
        &mut self,
        sys_ids: &[u32],
        range: Option<f32>,
        radius: f32,
        mode: ModeConfig,
        mmap_tree: bool,
    ) -> AstronavResult<(Duration, Vec<System>)> {
        self.interrupted.store(false, Ordering::SeqCst);
        if range.is_none() && self.ship.is_none() {
            return Err(AstronavError::RuntimeError(
                "Need either a jump range or a ship to compute a route with!"
                    .to_owned(),
            ));
        };
        let range = range.unwrap_or_else(|| {
            self.ship.as_ref().map_or_else(
                || unreachable!(),
                |ship| ship.jump_range(ship.fuel_mass, true),
            )
        });
        let id_map = self.get_systems_by_ids(sys_ids)?;
        let hops: Vec<System> =
            sys_ids.iter().map(|id| id_map[id].clone()).collect();
        let pos = hops.iter().map(|s| s.pos).collect_vec();
        self.load(mmap_tree, &pos, radius)?;
        let dt = Instant::now();
        let route = self.multiroute(&hops, range, mode)?;
        Ok((dt.elapsed(), route))
    }

    fn id(&self, sys: &TreeNode) -> AstronavResult<u32> {
        let id = self.tree.id(sys);
        if id == u32::MAX {
            let pos = sys.pos();
            Err(AstronavError::SystemNotFoundError(SysEntry::Pos(
                pos[0], pos[1], pos[2],
            )))
        } else {
            Ok(id)
        }
    }
}
