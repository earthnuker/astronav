use std::time::Instant;

use color_eyre::Result;
use human_repr::*;
use rustc_hash::FxHashMap;
use tracing::*;

use crate::{
    common::{dist, heuristic, MinFHeap, System},
    route::Router,
};
impl Router {
    pub(crate) fn route_incremental_broadening(
        &self,
        src: &System,
        dst: &System,
        range: f32,
    ) -> Result<Vec<System>> {
        /*
        h = (dist(node,goal)-(range*node.mult)).max(0.0) // remaining distance after jumping from here
        */
        let tree = self.tree.clone();
        let goal = tree.resolve(dst)?;
        let src = tree.resolve(src)?;
        let goal_id = tree.id(goal);
        let mut best_node = FxHashMap::default();
        // let mut prev = FxHashMap::default();
        let mut queue = MinFHeap::new();
        let t_start = Instant::now();
        let mut n = 0usize;
        let mut skipped = 0usize;
        let mut global_best = u32::MAX;
        queue.push(heuristic(range, src, goal, false), (0, src));
        loop {
            println!("Q: {}", queue.len());
            if queue.is_empty() {
                warn!(
                    "Visited: {} | Skipped: {} | search space exhausted after {}",
                    n.human_count_bare(),
                    skipped.human_count_bare(),
                    t_start.elapsed().human_duration()
                );
                break;
            }
            while let Some((_, (depth, node))) = queue.pop() {
                let best_len = best_node.len();
                let node_id = tree.id(node);
                let best_depth = best_node.entry(node_id).or_insert(depth);
                if *best_depth > global_best {
                    skipped += 1;
                    continue;
                }
                if depth < *best_depth {
                    *best_depth = depth;
                }
                n += 1;
                if node_id == goal_id {
                    if depth < global_best {
                        global_best = global_best.min(depth);
                        *queue = queue
                            .drain()
                            .filter(|(_, (d, _))| *d <= global_best)
                            .collect();
                        info!(
                            "Queued: {}, Skipped: {}, Seen: {} (Total: {}) | Best: {} | elapsed: {}",
                            queue.len().human_count_bare(),
                            skipped.human_count_bare(),
                            n.human_count_bare(),
                            best_len,
                            global_best,
                            t_start.elapsed().human_duration()
                        );
                    }
                    continue;
                } else if n % 10000 == 0 {
                    info!(
                        "Queued: {}, Skipped: {}, Seen: {} (Total: {}) | Best: {} | elapsed: {}",
                        queue.len().human_count_bare(),
                        skipped.human_count_bare(),
                        n.human_count_bare(),
                        best_len,
                        global_best,
                        t_start.elapsed().human_duration()
                    );
                }
                self.neighbours(node, range, |nb| {
                    let nb_id = tree.id(nb);
                    let ok = match best_node.get(&nb_id) {
                        Some(&d) => depth < d,
                        None => true,
                    };
                    if !ok {
                        return;
                    }
                    queue.push(dist(nb.pos(), goal.pos()), (depth + 1, nb));
                    return;
                });
            }
        }
        todo!()
    }
}
