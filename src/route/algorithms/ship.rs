use std::{collections::BinaryHeap, ops::ControlFlow, time::Instant};

use color_eyre::{eyre::bail, Result};
use human_repr::{HumanDuration, HumanThroughput};
use rustc_hash::FxHashMap;
use tracing::*;

use crate::{
    common::{dist, fcmp, System, TreeNode},
    route::{Router, ShipMode},
};

#[derive(Debug)]
struct ShipRouteState<'a> {
    cost: f32,
    fuel: f32,
    node: &'a TreeNode,
    refuels: usize,
    depth: usize,
    dist: f32,
    mode: ShipMode,
}

impl<'a> Ord for ShipRouteState<'a> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.mode != other.mode {
            panic!(
                "Trying to compare incompatible states: {:?} and {:?}",
                self.mode, other.mode
            );
        };
        match self.mode {
            ShipMode::Fuel => {
                // (cost,refuels)
                fcmp(self.cost, other.cost)
                    .then(self.refuels.cmp(&other.refuels))
            }
            ShipMode::Jumps => {
                // (depth,refuels,cost)
                self.depth
                    .cmp(&other.depth)
                    .then(self.refuels.cmp(&other.refuels))
                    .then(fcmp(self.cost, other.cost))
            }
        }
        .reverse()
    }
}

impl<'a> PartialOrd for ShipRouteState<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a> PartialEq for ShipRouteState<'a> {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::addr_eq(self.node, other.node)
            && self.depth == other.depth
            && self.refuels == other.refuels
    }
}

impl<'a> Eq for ShipRouteState<'a> {}

impl Router {
    pub(crate) fn route_ship(
        &mut self,
        start_sys: &System,
        goal_sys: &System,
        mode: &ShipMode,
    ) -> Result<Vec<System>> {
        let Some(ship) = self.ship.as_ref() else {
            bail!("Need a ship for computing fuel consumption!")
        };
        let t_start = Instant::now();
        let mut found = false;
        let mut num: usize = 0;
        let mut skipped: usize = 0;
        let mut unreachable: usize = 0;
        const INF: f32 = std::f32::INFINITY;
        let mut queue: BinaryHeap<ShipRouteState> = BinaryHeap::new();
        let mut best: FxHashMap<u32, f32> = FxHashMap::default();
        let mut prev: FxHashMap<u32, u32> = FxHashMap::default();
        let max_range = ship.max_range();
        let tree = self.get_tree(); // Assuming you have a method to get the RouterTree
        let start_node = start_sys.to_node(&tree)?;
        let goal_node = goal_sys.to_node(&tree)?;
        let state = ShipRouteState {
            cost: 0.0,
            fuel: ship.fuel_capacity,
            node: start_node,
            refuels: 0,
            depth: 0,
            dist: 0.0,
            mode: *mode,
        };
        queue.push(state);
        let mut last_new = Instant::now();
        while let Some(state) = queue.pop() {
            if state.node == goal_node {
                found = true;
                break;
            }
            if num % 100_000 == 0 {
                info!(
                    "D: ({}, {}) | FC: ({}, {}) | N: {} ({}) | B: {} ({}) | Q: {} | UR: {} | SK: {}",
                    state.depth,
                    state.dist,
                    state.refuels,
                    state.cost,
                    num,
                    prev.len(),
                    best.len(),
                    last_new.elapsed().human_duration(),
                    queue.len(),
                    unreachable,
                    skipped
                );
            }
            num += 1;
            let best_cost = best.get(&tree.id(state.node)).unwrap_or(&INF);
            if state.cost > *best_cost {
                skipped += 1;
                continue;
            }
            let mult = state.node.mult();
            self.neighbours(&state.node, max_range, |nb| {
                let nb_id = tree.id(nb);
                let mut refuels = state.refuels;
                let dist = dist(nb.pos(), state.node.pos());
                let (new_fuel, fuel_cost) = {
                    if let Some(res) =
                        ship.fuel_cost_for_jump(state.fuel, dist, mult)
                    {
                        // can jump with current amount of fuel
                        (state.fuel, res)
                    } else if let Some(res) =
                        ship.fuel_cost_for_jump(ship.fuel_capacity, dist, mult)
                    {
                        // can jump after refuel
                        refuels += 1;
                        (ship.fuel_capacity, res)
                    } else {
                        // can't jump
                        unreachable += 1;
                        return;
                    }
                };
                let next_cost = best.entry(nb_id).or_insert(INF);
                let new_cost = state.cost + fuel_cost;
                if new_cost < *next_cost {
                    last_new = Instant::now();
                    *next_cost = new_cost;
                    prev.insert(nb_id, tree.id(state.node));
                    queue.push(ShipRouteState {
                        cost: new_cost,
                        fuel: new_fuel - fuel_cost,
                        node: nb,
                        refuels,
                        depth: state.depth + 1,
                        dist: state.dist + dist,
                        mode: state.mode,
                    });
                }
                return;
            });
        }
        let rate = (prev.len() as f64) / t_start.elapsed().as_secs_f64();
        info!(
            "Took: {:.2}, {:.2}",
            t_start.elapsed().human_duration(),
            rate.human_throughput(" systems")
        );
        if !found {
            bail!("Search space exhausted");
        }
        Ok(self.reconstruct(goal_sys.id, &prev, &Default::default())?)
    }
}
