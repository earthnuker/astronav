use std::{ops::ControlFlow, time::Instant};

use color_eyre::{eyre::{bail, OptionExt}, Result};
use rustc_hash::FxHashMap;
use sif_kdtree::WithinDistance;
use tracing::warn;

use crate::{
    common::{dist, FormatFloat, MinFHeap, System, TreeNode, F32},
    event::{Event, RouteState},
    route::Router,
};

impl Router {
    pub(crate) fn route_astar(
        &self,
        src: &System,
        dst: &System,
        range: f32,
        weight: f32,
    ) -> Result<Vec<System>> {
        let src_name = src.name.clone();
        let dst_name = dst.name.clone();
        let start_sys = src;
        let goal_sys = dst;
        let d_total = dist(&start_sys.pos, &goal_sys.pos);
        let mut d_rem = d_total;
        let t_start = Instant::now();
        let tree = self.tree.clone();
        let mut state = RouteState {
            workers: 0,
            mode: self.mode,
            depth: 0,
            queue_size: 0,
            d_rem: d_total,
            d_total,
            prc_done: 0.0,
            n_seen: 0,
            prc_seen: 0.0,
            from: src_name,
            to: dst_name,
            system: start_sys.name.clone(),
            rate: 0.0,
            msg: None,
            refuels: None,
        };
        let total = self.tree.len() as f32;
        // let mut seen_v = vec![0u64;(self.tree.len()>>8)+1];
        let mut t_last = Instant::now();
        let mut prev = FxHashMap::default();
        let mut seen: FxHashMap<u32, f32> = FxHashMap::default();
        let mut found = false;
        let mut queue: MinFHeap<(usize, F32, &TreeNode)> = MinFHeap::new();
        let start_node = tree.resolve(&start_sys)?;
        let goal_node = tree.resolve(&goal_sys)?;
        queue.push(
            0.0,
            (
                0,        // depth,
                F32(0.0), // path_len
                start_node,
            ),
        );
        seen.insert(self.tree.id(start_node), 0.0);
        while !found {
            while let Some((_, (depth, path_len, node))) = queue.pop() {
                if t_last.elapsed() > self.status_interval {
                    let sys = node.get(self)?;
                    t_last = Instant::now();
                    state.depth = depth;
                    state.queue_size = queue.len();
                    state.prc_done = ((d_total - d_rem) * 100f32) / d_total;
                    state.d_rem = d_rem;
                    state.n_seen = seen.len();
                    state.prc_seen = ((seen.len() * 100) as f32) / total;
                    state.system = sys.name.clone();
                    state.rate =
                        (seen.len() as f64) / t_start.elapsed().as_secs_f64();
                    let path_len = (*path_len).format_float();
                    state.msg = Some(format!("Length: {path_len} Ly"));
                    self.emit(&Event::SearchState(state.clone()))?;
                }
                if tree.id(node) == self.tree.id(goal_node) {
                    queue.clear();
                    found = true;
                    break;
                }
                self.tree.look_up(
                    &WithinDistance::new(*node.pos(), range * node.mult()),
                    |nb| {
                        let nb_id = tree.id(nb);
                        if seen.contains_key(&nb_id) {
                            return ControlFlow::Continue(());
                        }
                        prev.insert(nb_id, node);
                        let d_g = nb.distp(goal_node);
                        d_rem = d_rem.min(d_g);
                        let path_len = (weight == 0.0)
                            .then_some(0.0)
                            .unwrap_or_else(|| {
                                weight.mul_add(
                                    dist(node.pos(), nb.pos()),
                                    *path_len,
                                )
                            });
                        let w = d_g + path_len;
                        let nw = seen.entry(nb_id).or_insert(w);
                        if w <= *nw {
                            queue.push(w, (depth + 1, F32(path_len), nb));
                            *nw = w;
                        }
                        ControlFlow::Continue(())
                    },
                );
            }
            if queue.is_empty() {
                break;
            }
        }
        if !found {
            bail!("Search space exhausted!");
        }
        let mut v: Vec<System> = Vec::new();
        let mut curr_sys = goal_sys.clone();
        loop {
            v.push(curr_sys.clone());
            match prev.get(&curr_sys.id) {
                Some(sys) => {
                    curr_sys = sys.get(self)?;
                }
                None => {
                    break;
                }
            }
        }
        v.reverse();
        Ok(v)
    }
}
