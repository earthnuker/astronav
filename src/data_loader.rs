use std::{
    num::TryFromIntError, path::Path, sync::Arc, thread::JoinHandle,
    time::Instant,
};

use color_eyre::eyre::{anyhow, Result};
use crossbeam_channel::{bounded, Sender};
use fs_err::File;
use itertools::Itertools;
use memmap2::Mmap;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use triple_accel::levenshtein::{levenshtein_simd_k_with_opts, RDAMERAU_COSTS};

use crate::common::{
    line_dist, AstronavError, AstronavResult, SystemFlags, TreeNode, F32,
};

pub(crate) const HEADER_SIZE: usize = 16;

struct DataLoader {
    mm_data: Arc<Mmap>,
    mm_names: Arc<Mmap>,
    mm_index: Arc<Mmap>,
}

type SearchWorkers =
    (Sender<Vec<SearchEntry>>, Vec<JoinHandle<Vec<SearchResult>>>);

struct SearchEntry {
    id: u32,
    offset: usize,
    len: usize,
}

#[derive(Clone)]
pub(crate) struct SearchResult {
    pub key: Vec<u8>,
    pub best_index: Option<u32>,
    hash: u64,
    pub best_dist_eudex: u32,
    pub best_dist_levenstein: u32,
    pub n_eudex: u64,
    pub n_lev: u64,
}

impl std::fmt::Debug for SearchResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SearchResult")
            .field("key", &std::str::from_utf8(&self.key).unwrap_or_default())
            .field("best_index", &self.best_index)
            .field("hash", &format!("{:08X}", &self.hash))
            .field("best_dist_eudex", &self.best_dist_eudex)
            .field("best_dist_levenstein", &self.best_dist_levenstein)
            .field("n_eudex", &self.n_eudex)
            .field("n_lev", &self.n_lev)
            .finish()
    }
}

impl Default for SearchResult {
    fn default() -> Self {
        Self {
            key: vec![],
            best_index: None,
            hash: u64::MAX,
            best_dist_eudex: u32::MAX,
            best_dist_levenstein: u32::MAX,
            n_eudex: 0,
            n_lev: 0,
        }
    }
}

impl SearchResult {
    pub(crate) fn new(s: &[u8]) -> Self {
        Self { key: s.to_owned(), hash: eudex_hash(s), ..Default::default() }
    }

    pub(crate) fn update(&mut self, idx: u32, name: &[u8]) {
        if self.best_dist_levenstein == 0 {
            return;
        }
        self.n_eudex += 1;
        let eudex_dist = eudex_dist(self.hash, eudex_hash(name));
        if eudex_dist <= self.best_dist_eudex {
            self.best_dist_eudex = eudex_dist;
            let dist_levenstein =
                triple_accel::levenshtein_exp(&self.key, name);
            self.n_lev += 1;
            if dist_levenstein < self.best_dist_levenstein {
                self.best_dist_levenstein = dist_levenstein;
                self.best_index = Some(idx);
            }
        }
    }

    pub(crate) fn merge(&mut self, other: &Self) {
        self.n_eudex += other.n_eudex;
        self.n_lev += other.n_lev;
        if other.best_dist_eudex <= self.best_dist_eudex
            && other.best_dist_levenstein < self.best_dist_levenstein
        {
            self.key = other.key.clone();
            self.best_dist_eudex = other.best_dist_eudex;
            self.best_dist_levenstein = other.best_dist_levenstein;
            self.best_index = other.best_index;
        }
    }
}

const fn eudex_dist(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

fn eudex_hash(string: &[u8]) -> u64 {
    // from https://docs.rs/eudex/latest/src/eudex/lib.rs.html#35
    let mut b = 0;
    let first_byte =
        eudex::raw::map_first(*string.first().unwrap_or(&0)) as u64;

    let mut res = 0;
    let mut n = 1u8;

    loop {
        b += 1;
        // Detect overflows into the first slot.
        if n == 0 || b >= string.len() {
            break;
        }

        if let Some(x) = eudex::raw::filter(res as u8, string[b]) {
            res <<= 8;
            res |= x as u64;
            n <<= 1;
        }
    }

    res | (first_byte << 56)
}

pub(crate) struct MappedNodes {
    // TODO: Use two separate lists for positions and flags?
    pub nodes: Box<[TreeNode]>,
    _mm: Arc<Mmap>,
}

impl Drop for MappedNodes {
    fn drop(&mut self) {
        let _ = Box::leak(std::mem::take(&mut self.nodes));
    }
}

impl std::ops::Deref for MappedNodes {
    type Target = [TreeNode];

    fn deref(&self) -> &Self::Target {
        &self.nodes
    }
}

impl AsRef<[TreeNode]> for MappedNodes {
    fn as_ref(&self) -> &[TreeNode] {
        &self.nodes
    }
}

impl DataLoader {
    fn new<P: AsRef<Path>>(base_path: P) -> Result<Self> {
        let base_path = base_path.as_ref();
        let data_file = base_path.with_extension("bin");
        let names_file = base_path.with_extension("names");
        let index_file = base_path.with_extension("idx");
        Ok(Self {
            mm_data: Arc::new(unsafe { Mmap::map(&File::open(data_file)?) }?),
            mm_names: Arc::new(unsafe { Mmap::map(&File::open(names_file)?) }?),
            mm_index: Arc::new(unsafe { Mmap::map(&File::open(index_file)?) }?),
        })
    }

    fn min_dist_hops(hops: &[[f32; 3]], pos: [f32; 3]) -> f32 {
        hops.iter()
            .tuple_windows()
            .map(|(a, b)| F32(line_dist(&pos, a, b)))
            .min()
            .map_or(f32::MAX, |v| *v)
    }

    fn map(&self) -> (MappedNodes, usize) {
        let total_nodes = (self.mm_data.len() - HEADER_SIZE)
            / std::mem::size_of::<TreeNode>();
        let mm = Arc::clone(&self.mm_data);

        let nodes = unsafe {
            Vec::from_raw_parts(
                mm[HEADER_SIZE..].as_ptr().cast::<TreeNode>().cast_mut(),
                total_nodes,
                total_nodes,
            )
            .into_boxed_slice()
        };
        (MappedNodes { _mm: mm, nodes }, total_nodes)
    }

    fn load(&self, hops: &[[f32; 3]], dist: f32) -> (Box<[TreeNode]>, usize) {
        let total_nodes = (self.mm_data.len() - HEADER_SIZE)
            / std::mem::size_of::<TreeNode>();
        let nodes_vec = unsafe {
            std::slice::from_raw_parts(
                self.mm_data[HEADER_SIZE..].as_ptr().cast::<TreeNode>(),
                total_nodes,
            )
        };
        let mut nodes = Vec::with_capacity(total_nodes);
        nodes_vec.into_par_iter().cloned().collect_into_vec(&mut nodes);
        (nodes.into_boxed_slice(), total_nodes)
    }

    fn spawn_fuzzy_search_workers(
        &self,
        query: &[SearchResult],
    ) -> SearchWorkers {
        let query = query.to_owned();
        let (tx, rx) = bounded::<Vec<SearchEntry>>(4096);
        let workers = (0..(num_cpus::get()))
            .map(|_| {
                let mut query = query.clone();
                let rx = rx.clone();
                let mm = self.mm_names.clone();
                std::thread::spawn(move || {
                    rx.into_iter()
                        .flat_map(std::iter::IntoIterator::into_iter)
                        .for_each(|entry| {
                            for query in query.iter_mut() {
                                query.update(
                                    entry.id,
                                    &mm[entry.offset..entry.offset + entry.len],
                                );
                            }
                        });
                    query
                })
            })
            .collect_vec();
        (tx, workers)
    }

    fn search<S: AsRef<str>>(
        &self,
        query: &[S],
    ) -> Result<(f64, FxHashMap<String, SearchResult>), AstronavError> {
        let query_len = query.len();
        let mut query = query
            .iter()
            .map(|s| SearchResult::new(s.as_ref().as_bytes()))
            .collect_vec();
        let (tx, workers) = self.spawn_fuzzy_search_workers(&query);
        let addr = self.mm_names.as_ptr();
        let t_start = Instant::now();
        self.mm_names
            .split(|&c| c == b'\n')
            .skip(1)
            .enumerate()
            .map(|(sys_id, name)| {
                let sys_id = sys_id.try_into().unwrap_or_else(|e| unreachable!("{e}: Somehow you got more than 2^32 star systems, congratulations!"));
                let name_ptr = name.as_ptr();
                let name_offset = unsafe { name_ptr.offset_from(addr) };
                let name_offset = name_offset.try_into().unwrap_or_else(|e| unreachable!("Invalid offset {name_offset}: {e}"));
                let name_len = name.len();
                SearchEntry {
                    id: sys_id,
                    offset: name_offset,
                    len: name_len,
                }
            })
            .chunks(4096)
            .into_iter()
            .try_for_each(|chunk| tx.send(chunk.collect_vec()))
            .map_err(|e| AstronavError::Other(e.into()))?;
        drop(tx);
        for w in workers {
            let w = w.join().map_err(|err| {
                AstronavError::Other(anyhow!("failed to join thread: {err:?}"))
            })?;
            for (query, res) in query.iter_mut().zip(w) {
                query.merge(&res);
            }
        }
        let mut res = FxHashMap::default();
        res.reserve(query.len());
        for query in query {
            let key = std::str::from_utf8(&query.key)
                .map_err(|e| AstronavError::Other(e.into()))?
                .to_owned();
            res.insert(key, query);
        }
        let num_systems = (self.mm_data.len() - HEADER_SIZE)
            / std::mem::size_of::<TreeNode>();
        let rate = ((num_systems * query_len) as f64)
            / t_start.elapsed().as_secs_f64();
        Ok((rate, res))
    }

    fn get_by_id64s(
        &self,
        ids: &[u64],
    ) -> AstronavResult<(f64, FxHashMap<u64, u32>)> {
        let num_systems = (self.mm_index.len() - 16) / (8 + 4);
        let data: &[[u8; 8 + 4]] = unsafe {
            std::slice::from_raw_parts(
                self.mm_index[HEADER_SIZE..].as_ptr().cast::<_>(),
                num_systems,
            )
        };

        let t_start = Instant::now();
        let ret: AstronavResult<FxHashMap<u64, u32>> = ids
            .iter()
            .map(|&id| {
                data.par_iter()
                    .position_any(|res| {
                        let mut buf = [0u8; 8];
                        buf.copy_from_slice(&res[..8]);
                        id == u64::from_le_bytes(buf)
                    })
                    .ok_or_else(|| {
                        AstronavError::SystemNotFoundError(
                            crate::common::SysEntry::ID64(id),
                        )
                    })
                    .and_then(|n| {
                        Ok((
                            id,
                            n.try_into().map_err(|e: TryFromIntError| {
                                AstronavError::Other(e.into())
                            })?,
                        ))
                    })
            })
            .collect();
        let rate = ((num_systems * ids.len()) as f64)
            / t_start.elapsed().as_secs_f64();
        Ok((rate, ret?))
    }
}

pub fn search<P: AsRef<Path>, S: AsRef<str>>(
    path: P,
    query: &[S],
) -> Result<(f64, FxHashMap<String, SearchResult>), AstronavError> {
    DataLoader::new(path)?.search(query)
}

pub fn load<P: AsRef<Path>>(
    path: P,
    hops: &[[f32; 3]],
    dist: f32,
) -> Result<(Box<[TreeNode]>, usize), AstronavError> {
    Ok(DataLoader::new(path)?.load(hops, dist))
}

pub fn map<P: AsRef<Path>>(
    path: P,
) -> Result<(MappedNodes, usize), AstronavError> {
    Ok(DataLoader::new(path)?.map())
}

pub fn get_by_id64<P: AsRef<Path>>(
    path: P,
    id64s: &[u64],
) -> Result<(f64, FxHashMap<u64, u32>), AstronavError> {
    DataLoader::new(path)?.get_by_id64s(id64s)
}
