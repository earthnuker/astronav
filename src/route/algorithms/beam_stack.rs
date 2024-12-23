use std::{collections::BTreeMap, ops::ControlFlow};

use color_eyre::Result;
use rustc_hash::FxHashMap;

use crate::{
    common::{heuristic, MinFHeap, System, TreeNode},
    route::Router,
};

impl Router {
    pub(crate) fn route_beam_stack(
        &self,
        src: &System,
        dst: &System,
        range: f32,
    ) -> Result<Vec<System>> {
        // https://www.youtube.com/watch?v=OhBpJEmmflQ
        let tree = self.tree.clone();
        todo!();
        let goal = tree.resolve(dst)?;
        let mut queue_stack: BTreeMap<usize, MinFHeap<&TreeNode>> =
            BTreeMap::new();
        // let mut prev = FxHashMap::default();
        let mut seen: FxHashMap<u32, usize> = FxHashMap::default();
        let src = tree.resolve(src)?;
        let mut steps = 0usize;
        let mut skipped = 0usize;
        let mut best_depth = usize::MAX;
        seen.insert(tree.id(src), 0);
        let mut heap = MinFHeap::default();
        heap.push(heuristic(range, src, &goal, false), src);
        queue_stack.insert(0, heap);
        while let Some(depth) = queue_stack.last_key_value().map(|(k, _)| *k) {
            let Some(queue) = queue_stack.get_mut(&depth) else {
                continue;
            };
            if queue.is_empty() {
                queue_stack.remove(&depth);
                continue;
            }
            if let Some((_, node)) = queue.pop() {
                self.neighbours(node, range, |nb| {});
            }
        }
        todo!();
    }
}
