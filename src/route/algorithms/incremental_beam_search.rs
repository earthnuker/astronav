use std::{
    cmp::Reverse,
    collections::{BTreeMap, VecDeque},
    ops::ControlFlow,
    sync::{Arc, RwLock},
    time::Instant,
};

use bitvec_simd::BitVec;
use color_eyre::{eyre::bail, Result};
use crossbeam_channel::bounded;
use human_repr::{HumanDuration, HumanThroughput};
use itertools::*;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use sif_kdtree::WithinDistance;
use tracing::*;

/*
while True:
    node=next_node()
    nb=get_neighbors(node)
    nb.sort_by_hueristic()
    next, rest = nb.split_at(bw)
    queue_next.extend(next)
    stash[depth].extent(rest)
*/
use crate::{
    common::{
        dist, heuristic, sort_by_heuristic, MinFHeap, System, TreeNode, F32,
    },
    event::{Event, RouteState},
    route::Router,
};

struct State<'a> {
    goal: &'a TreeNode,
    seen: FxHashMap<u32, usize>,
    route_state: RouteState,
    range: f32,
    beam_width: usize,
    best: usize,
    prev: FxHashMap<u32, u32>,
    stash: BTreeMap<usize, Vec<u32>>,
}

impl Router {
    fn route_ibs_inner<'a>(
        &self,
        current: &[&TreeNode],
        state: &mut State<'_>,
        depth: usize,
    ) {
        let tree = self.tree.clone();
        let goal_id = tree.id(state.goal);
        let mut nbs: Vec<u32> = Vec::new();
        if depth > state.best {
            return;
        }
        for node in current {
            let node_id = tree.id(node);
            if node_id == tree.id(state.goal) {
                state.best = depth;
                info!("New best: {}", state.best);
                return;
            }
            let node = tree[node_id as usize];
            tree.look_up(
                &WithinDistance::new(*node.pos(), state.range * node.mult()),
                |nb| {
                    let nb_id = self.tree.id(nb);
                    if depth
                        >= state.seen.get(&nb_id).copied().unwrap_or(usize::MAX)
                    {
                        return ControlFlow::Continue(());
                    }
                    state.seen.insert(nb_id, depth);
                    state.prev.insert(nb_id, node_id);
                    nbs.push(nb_id);
                    return ControlFlow::Continue(());
                },
            );
        }
        let nodes = state.stash.entry(depth).or_default();
        nodes.extend(nbs);
        let mut nodes = nodes.drain(..).map(|id| &tree[id as usize]).collect_vec();
        sort_by_heuristic(&mut nodes, &state.goal, state.range, true);
        let next =
            nodes.drain(..state.beam_width.min(nodes.len())).collect_vec();
        self.route_ibs_inner(&next, state, depth + 1);
    }

    pub(crate) fn route_ibs(
        &self,
        start_sys: &System,
        goal_sys: &System,
        range: f32,
        beam_width: usize,
    ) -> Result<Vec<System>> {
        let mut state = State {
            goal: self.tree.resolve(goal_sys)?,
            seen: FxHashMap::default(),
            route_state: self.initial_state(start_sys, goal_sys, 0),
            range,
            beam_width,
            best: usize::MAX,
            prev: FxHashMap::default(),
            stash: BTreeMap::default(),
        };
        let node = self.tree.resolve(&start_sys)?;
        loop {
            self.route_ibs_inner(&[node], &mut state, 0);
            // for (k, v) in state.stash.iter() {
            //     println!("{k}: {l}", l = v.len());
            // }
        }
        todo!()
    }
}
