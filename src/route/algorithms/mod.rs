use std::fmt::Display;

use color_eyre::Result;
use parse_display::Display;

use super::Router;
use crate::common::{BeamWidth, System, TreeNode, F32};

mod astar;
mod beam;
mod beam_stack;
mod dfs;
mod dijstra;
mod incremental_beam_search;
mod incremental_broadening;
mod mcts;
mod ship;
