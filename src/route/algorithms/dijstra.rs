use color_eyre::Result;

use crate::{common::System, route::Router};

impl Router {
    pub fn route_dijkstra(
        &self,
        src: &System,
        dst: &System,
        range: f32,
    ) -> Result<Vec<System>> {
        self.route_astar(src, dst, range, 0.0)
    }
}
