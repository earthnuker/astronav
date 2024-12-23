use core::f32;
use std::{
    cmp::Reverse,
    collections::BinaryHeap,
    ops::ControlFlow,
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc,
    },
    time::Instant,
};

use bitvec_simd::BitVec;
use color_eyre::{eyre::bail, Result};
use crossbeam_channel::bounded;
use human_repr::*;
use itertools::*;
use parking_lot::RwLock;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use sif_kdtree::WithinDistance;
use sorted_vec::SortedVec;
use tracing::*;

use crate::{
    common::{
        dist, heuristic, AstronavError, BeamWidth, MinFHeap, StarKind, System,
        TreeNode, F32,
    },
    event::{Event, RouteState},
    route::{RefuelMode, Router},
};

/// Implements the `Router` struct with methods for beam search routing.
///
/// # Methods
///
/// - `route_beam_impl`: A generic function that performs the beam search
///   routing algorithm.
///   - `HAS_REFUEL`: A constant indicating if refueling is considered.
///   - `HAS_SHIP`: A constant indicating if a ship is present.
///   - `HAS_LIMIT`: A constant indicating if there is a range limit.
///   - `NEEDS_SORT`: A constant indicating if sorting is needed.
///   - Parameters:
///     - `start_sys`: The starting system.
///     - `goal_sys`: The goal system.
///     - `range`: The range of the ship.
///     - `beam_width`: The width of the beam.
///     - `refuel_mode`: The refuel mode.
///     - `refuel_primary`: A boolean indicating if refueling at primary stars
///       is allowed.
///     - `boost_primary`: A boolean indicating if boosting at primary stars is
///       allowed.
///     - `range_limit`: The range limit.
///   - Returns: A `Result` containing a vector of systems representing the
///     route.
///
/// - `route_beam`: A function that determines the appropriate template
///   parameters and calls `route_beam_impl`.
///   - Parameters:
///     - `start_sys`: The starting system.
///     - `goal_sys`: The goal system.
///     - `range`: The range of the ship.
///     - `beam_width`: The width of the beam.
///     - `refuel_mode`: The refuel mode.
///     - `refuel_primary`: A boolean indicating if refueling at primary stars
///       is allowed.
///     - `boost_primary`: A boolean indicating if boosting at primary stars is
///       allowed.
///     - `range_limit`: The range limit.
///   - Returns: A `Result` containing a vector of systems representing the
///     route.
impl Router {
    /// Performs the beam search routing algorithm.
    ///
    /// # Parameters
    ///
    /// - `HAS_REFUEL`: A constant indicating if refueling is considered.
    /// - `HAS_SHIP`: A constant indicating if a ship is present.
    /// - `HAS_LIMIT`: A constant indicating if there is a range limit.
    /// - `NEEDS_SORT`: A constant indicating if sorting is needed.
    /// - `start_sys`: The starting system.
    /// - `goal_sys`: The goal system.
    /// - `range`: The range of the ship.
    /// - `beam_width`: The width of the beam.
    /// - `refuel_mode`: The refuel mode.
    /// - `refuel_primary`: A boolean indicating if refueling at primary stars
    ///   is allowed.
    /// - `boost_primary`: A boolean indicating if boosting at primary stars is
    ///   allowed.
    /// - `range_limit`: The range limit.
    ///
    /// # Returns
    ///
    /// A `Result` containing a vector of systems representing the route.
    #[allow(clippy::too_many_arguments)]
    #[inline(never)]
    pub(crate) fn route_beam_impl<
        const HAS_REFUEL: bool,
        const HAS_SHIP: bool,
        const HAS_LIMIT: bool,
        const NEEDS_SORT: bool,
    >(
        &self,
        (
            start_sys,
            goal_sys,
            mut range,
            beam_width,
            refuel_mode,
            refuel_primary,
            boost_primary,
            range_limit,
        ): (
            &System,
            &System,
            f32,
            &BeamWidth,
            Option<&RefuelMode>,
            bool,
            bool,
            f32,
        ),
    ) -> Result<Vec<System>> {
        let range_limit = HAS_LIMIT.then_some(range_limit);
        if HAS_SHIP {
            if let Some(ship) = self.ship.as_ref() {
                range = ship.max_range();
            }
        }
        let mut refuel_mode = refuel_mode;
        let (fuel_capacity, min_fuel) = self
            .ship
            .as_ref()
            .map(|s| (s.fuel_capacity, s.fsd.max_fuel))
            .unwrap_or((f32::NAN, f32::NAN));
        let refuel_every_jump =
            matches!(refuel_mode, Some(RefuelMode::WhenPossible));
        let refuel_amount = match refuel_mode {
            Some(RefuelMode::WhenPossible) => fuel_capacity,
            Some(RefuelMode::WhenEmpty) => fuel_capacity,
            Some(RefuelMode::LeastJumps) => min_fuel,
            None => fuel_capacity,
        };
        if self.ship.is_none() && refuel_mode.is_some() {
            warn!("No ship loaded, ignoring refueling mode");
            refuel_mode = None;
        }
        let num_workers = rayon::current_num_threads();
        info!("Running with {} worker(s)", num_workers);
        let t_start = Instant::now();
        let mut t_last = Instant::now();
        let mut seen = Arc::new(BitVec::zeros(self.max_id));
        let mut seen_update = BitVec::zeros(self.max_id);
        let mut prev = FxHashMap::default();
        let mut refuels = FxHashSet::default();
        // prev.reserve(self.tree.len());
        let src_name = start_sys.name.clone();
        let dst_name = goal_sys.name.clone();
        let mut queue: Vec<(f32, &TreeNode)> = Vec::new();
        let goal_id = goal_sys.id;
        let goal_node = goal_sys.to_node(&self.tree)?;
        if start_sys.id == goal_sys.id {
            return Ok(vec![goal_sys.clone()]);
        }
        let mut goal_reached = false;
        let tree = self.tree.clone();
        let total = self.tree.len() as f32;
        let d_total = dist(&start_sys.pos, &goal_sys.pos);
        if let BeamWidth::Absolute(width) = beam_width {
            queue.reserve(*width);
            prev.reserve(((*width as f32) * (d_total / range)) as usize);
        }
        let mut d_rem = d_total;
        let mut depth = 0;
        let mut state = RouteState {
            mode: self.mode,
            workers: num_workers,
            depth: 0,
            queue_size: 0,
            d_rem,
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
        let start_node = tree.resolve(start_sys)?;
        Arc::get_mut(&mut seen)
            .unwrap_or_else(|| unreachable!())
            .set(tree.id(start_node) as usize, true);
        queue.push((fuel_capacity, start_node));
        let mut queries = 0usize;
        let mut interrupted = false;
        // TODO?: replace with RwLocked buffer/BTreeSet?
        let (tx, rx) = bounded::<(u32, bool, Vec<(f32, f32, u32)>)>(1024);
        loop {
            if self.is_interrupted() {
                interrupted = true;
                break;
            }
            if goal_reached || queue.is_empty() {
                break;
            }
            let worker_queue = std::mem::take(&mut queue);
            let ship = refuel_mode.as_ref().and_then(|_| self.ship.clone());
            std::thread::scope(|s| {
                let worker_env =
                    (tree.clone(), tx.clone(), ship.clone(), seen.clone());
                let worker_thread = s.spawn(move || {
                    worker_queue.into_par_iter().for_each_with(
                        worker_env,
                        |(tree, tx, ship, seen), (mut fuel_available, node)| {
                            let mut needs_refuel = false;
                            if HAS_REFUEL {
                                let mut can_refuel = node.flags.primary_kind()
                                    == StarKind::Scoopable;
                                if !refuel_primary {
                                    can_refuel |=
                                        node.flags.kind() == StarKind::Scoopable
                                };
                                needs_refuel = ship.is_some()
                                    && (fuel_available < min_fuel
                                        || refuel_every_jump)
                                    && can_refuel;
                                if needs_refuel {
                                    fuel_available = refuel_amount;
                                };
                            }
                            let mut range = if HAS_SHIP {
                                ship.as_ref()
                                    .unwrap_or_else(|| unreachable!())
                                    .jump_range(fuel_available, true)
                            } else {
                                range
                            };
                            if HAS_LIMIT {
                                if let Some(range_limit) = range_limit {
                                    range *= range_limit;
                                }
                            }
                            let mult = if boost_primary {
                                node.primary_mult()
                            } else {
                                node.mult()
                            };
                            let mut buffer = Vec::with_capacity(1024);
                            tree.look_up(
                                &WithinDistance::new(*node.pos(), range * mult),
                                |nb| {
                                    if seen
                                        .get(tree.id(nb) as usize)
                                        .unwrap_or(false)
                                    {
                                        return ControlFlow::Continue(());
                                    }
                                    let fuel_available: Option<f32> =
                                        if HAS_SHIP {
                                            ship.as_ref()
                                                .unwrap_or_else(
                                                    || unreachable!(),
                                                )
                                                .fuel_cost_for_jump(
                                                    fuel_available,
                                                    dist(node.pos(), nb.pos()),
                                                    mult,
                                                )
                                                .map(|fuel_cost| {
                                                    fuel_available - fuel_cost
                                                })
                                        } else {
                                            Some(fuel_available)
                                        };
                                    if let Some(fuel_available) = fuel_available
                                    {
                                        buffer.push((
                                            0.0,
                                            fuel_available,
                                            tree.id(nb),
                                        ));
                                    }
                                    return ControlFlow::Continue(());
                                },
                            );
                            if let Err(e) =
                                tx.send((tree.id(node), needs_refuel, buffer))
                            {
                                panic!(
                                    "Failed to send edge to worker thread: {e}"
                                );
                            }
                        },
                    )
                });
                while !worker_thread.is_finished() {
                    for (parent_node_id, refuel, child_node_ids) in
                        rx.try_iter()
                    {
                        if goal_reached {
                            break;
                        }
                        for (_, fuel_remaining, node_id) in child_node_ids {
                            queries += 1;
                            if !seen_update
                                .get(node_id as usize)
                                .unwrap_or_default()
                            {
                                seen_update.set(node_id as usize, true);
                                prev.insert(node_id, parent_node_id);
                                if refuel {
                                    refuels.insert(parent_node_id);
                                }
                                let node = &tree[node_id as usize];
                                d_rem = d_rem.min(node.distp(goal_node));
                                if node_id == goal_id {
                                    goal_reached = true;
                                }
                                queue.push((fuel_remaining, node));
                            }
                        }
                    }
                }
            });
            depth += 1;
            if beam_width.needs_sort() && !queue.is_empty() {
                let bw = beam_width
                    .compute(queue.len())
                    .max(1)
                    .min(queue.len().saturating_sub(1));
                if self.deterministic {
                    queue.par_sort_by_key(|(_, node)| {
                        F32(heuristic(range, node, goal_node, boost_primary))
                    });
                } else {
                    queue.select_nth_unstable_by_key(bw, |(_, node)| {
                        F32(heuristic(range, node, goal_node, boost_primary))
                    });
                }
                queue.truncate(bw);
            } else if self.deterministic {
                queue.par_sort_by_key(|(_, node)| {
                    F32(heuristic(range, node, goal_node, boost_primary))
                });
            }
            Arc::get_mut(&mut seen)
                .unwrap_or_else(|| unreachable!())
                .or_inplace(&seen_update);
            if t_last.elapsed() > self.status_interval {
                t_last = Instant::now();
                let Some(cur_node) = queue.first() else {
                    continue;
                };
                let num_seen = seen.count_ones();
                state.depth = depth;
                state.queue_size = queue.len();
                state.prc_done = ((d_total - d_rem) * 100f32) / d_total;
                state.d_rem = d_rem;
                state.n_seen = num_seen;
                state.prc_seen = ((num_seen * 100) as f32) / total;
                state.rate =
                    (num_seen as f64) / t_start.elapsed().as_secs_f64();
                state.system = self.get(tree.id(cur_node.1))?.name;
                if !refuels.is_empty() {
                    state.refuels = Some(refuels.len());
                }
                state.msg = ship.map(|mut ship| {
                    ship.fuel_mass = cur_node.0;
                    format!("{ship}", ship = ship.get_inner())
                });
                self.emit(&Event::SearchState(state.clone()))?;
            }
        }
        let rate = (seen.count_ones() as f64) / t_start.elapsed().as_secs_f64();
        let query_rate = (queries as f64) / t_start.elapsed().as_secs_f64();
        let mut goal_id = goal_sys.id;
        if interrupted {
            warn!("Router computation interrupted, returning partial result!");
            goal_reached = true;
            goal_id = self
                .tree
                .iter()
                .filter(|&v| seen.get(tree.id(v) as usize).unwrap_or_default())
                .min_by_key(|node| {
                    F32(heuristic(range, node, goal_node, boost_primary))
                })
                .map(|node| tree.id(node))
                .ok_or_else(|| {
                    AstronavError::RuntimeError(
                        "No best node found!".to_owned(),
                    )
                })?;
        };
        info!(
            "Took: {:.2}, {:.2}, {:.2}",
            t_start.elapsed().human_duration(),
            rate.human_throughput(" systems"),
            query_rate.human_throughput(" queries")
        );
        if !goal_reached {
            bail!("Search space exhausted");
        }
        Ok(self.reconstruct(goal_id, &prev, &refuels)?)
    }

    /// Determines the appropriate template parameters and calls
    /// `route_beam_impl`.
    ///
    /// # Parameters
    ///
    /// - `start_sys`: The starting system.
    /// - `goal_sys`: The goal system.
    /// - `range`: The range of the ship.
    /// - `beam_width`: The width of the beam.
    /// - `refuel_mode`: The refuel mode.
    /// - `refuel_primary`: A boolean indicating if refueling at primary stars
    ///   is allowed.
    /// - `boost_primary`: A boolean indicating if boosting at primary stars is
    ///   allowed.
    /// - `range_limit`: The range limit.
    ///
    /// # Returns
    ///
    /// A `Result` containing a vector of systems representing the route.
    pub(crate) fn route_beam(
        &self,
        start_sys: &System,
        goal_sys: &System,
        range: f32,
        beam_width: &BeamWidth,
        refuel_mode: Option<&RefuelMode>,
        refuel_primary: bool,
        boost_primary: bool,
        range_limit: f32,
    ) -> Result<Vec<System>> {
        let limit_valid = range_limit > 0.0 && range_limit < 1.0;
        let has_refuel = refuel_mode.is_some();
        let has_ship = self.ship.is_some() && has_refuel;
        let beam_width_needs_sort = beam_width.needs_sort();
        let args = (
            start_sys,
            goal_sys,
            range,
            beam_width,
            refuel_mode,
            refuel_primary,
            boost_primary,
            range_limit,
        );
        match (has_refuel, has_ship, limit_valid, beam_width_needs_sort) {
            (false, false, false, false) => {
                self.route_beam_impl::<false, false, false, false>(args)
            }
            (false, false, false, true) => {
                self.route_beam_impl::<false, false, false, true>(args)
            }
            (false, false, true, false) => {
                self.route_beam_impl::<false, false, true, false>(args)
            }
            (false, false, true, true) => {
                self.route_beam_impl::<false, false, true, true>(args)
            }
            (false, true, false, false) => {
                self.route_beam_impl::<false, true, false, false>(args)
            }
            (false, true, false, true) => {
                self.route_beam_impl::<false, true, false, true>(args)
            }
            (false, true, true, false) => {
                self.route_beam_impl::<false, true, true, false>(args)
            }
            (false, true, true, true) => {
                self.route_beam_impl::<false, true, true, true>(args)
            }
            (true, false, false, false) => {
                self.route_beam_impl::<true, false, false, false>(args)
            }
            (true, false, false, true) => {
                self.route_beam_impl::<true, false, false, true>(args)
            }
            (true, false, true, false) => {
                self.route_beam_impl::<true, false, true, false>(args)
            }
            (true, false, true, true) => {
                self.route_beam_impl::<true, false, true, true>(args)
            }
            (true, true, false, false) => {
                self.route_beam_impl::<true, true, false, false>(args)
            }
            (true, true, false, true) => {
                self.route_beam_impl::<true, true, false, true>(args)
            }
            (true, true, true, false) => {
                self.route_beam_impl::<true, true, true, false>(args)
            }
            (true, true, true, true) => {
                self.route_beam_impl::<true, true, true, true>(args)
            }
        }
    }
}
