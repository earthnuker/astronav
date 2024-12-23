use color_eyre::Result;

use crate::{
    common::{BeamWidth, System},
    route::{RefuelMode, Router},
};

impl Router {
    pub fn route_mcts(
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
        let estimate_cost = |sys: &System| -> usize {
            dbg!(sys, beam_width);
            self.route_beam(
                sys,
                goal_sys,
                range,
                beam_width,
                refuel_mode,
                refuel_primary,
                boost_primary,
                range_limit,
            )
            .map(|r| r.len())
            .unwrap_or(usize::MAX)
        };
        dbg!(estimate_cost(start_sys));
        todo!()
    }
}
