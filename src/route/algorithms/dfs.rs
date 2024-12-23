use color_eyre::Result;

use crate::{
    common::{BeamWidth, System},
    route::Router,
};

impl Router {
    pub(crate) fn route_dfs(
        &self,
        start_sys: &System,
        goal_sys: &System,
        range: f32,
    ) -> Result<Vec<System>> {
        self.route_beam(
            start_sys,
            goal_sys,
            range,
            &BeamWidth::Absolute(1),
            None,
            false,
            false,
            1.0,
        )
    }
}
