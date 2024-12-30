use serde::{Deserialize, Serialize};
use strum::{Display, EnumDiscriminants, EnumIter, EnumVariantNames, VariantNames};

use crate::common::{BeamWidth, TreeNode, F32};

const fn default_weight() -> F32 {
    F32(1.0)
}

#[derive(
    Debug, Default, Eq, PartialEq, Copy, Clone, Deserialize, Serialize,
)]
#[serde(rename_all = "snake_case")]
pub enum ShipMode {
    #[default]
    Fuel,
    Jumps,
}

#[derive(
    Debug, Deserialize, Serialize, Copy, Clone, PartialEq, Eq, Display,
)]
#[serde(rename_all = "snake_case")]
pub enum RefuelMode {
    #[strum(serialize = "refueling when possible")]
    WhenPossible,
    #[strum(serialize = "refueling when empty")]
    WhenEmpty,
    #[strum(
        serialize = "refueling for least amount of jumps (partially fill tank)"
    )]
    LeastJumps,
}

//

#[derive(Debug, Deserialize, Serialize, Copy, Clone, EnumDiscriminants)]
#[strum_discriminants(name(RouteMode))]
#[strum_discriminants(derive(EnumIter, VariantNames, Display))]
#[strum_discriminants(strum(serialize_all = "title_case"))]
#[serde(rename_all = "snake_case", tag = "mode", deny_unknown_fields)]
pub enum ModeConfig {
    #[serde(alias = "beam")]
    #[strum_discriminants(strum(serialize = "Beam Search"))]
    BeamSearch {
        #[serde(default)]
        beam_width: BeamWidth,
        #[serde(default)]
        refuel_mode: Option<RefuelMode>,
        #[serde(default)]
        refuel_primary: bool,
        #[serde(default)]
        boost_primary: bool,
        #[serde(default)]
        range_limit: f32,
    },

    #[serde(alias = "dfs")]
    #[strum_discriminants(strum(serialize = "Depth-first Search"))]
    DepthFirst,

    #[serde(alias = "ib")]
    #[strum_discriminants(strum(
        serialize = "Incrementaly broadening Beam Search"
    ))]
    IncrementalBroadening, // TODO: implement IncrementalBroadening

    #[serde(alias = "bss")]
    #[strum_discriminants(strum(serialize = "Beam Stack search"))]
    BeamStack, // TODO: implement BeamStack

    #[serde(alias = "astar")]
    #[serde(alias = "a-star")]
    #[strum_discriminants(strum(serialize = "A*-Search"))]
    AStar {
        #[serde(default = "default_weight")]
        weight: F32,
    },

    #[strum_discriminants(strum(serialize = "Dijkstra's Algorithm"))]
    Dijkstra,

    #[serde(alias = "ibs")]
    #[strum_discriminants(strum(serialize = "Incremental Beam Search"))]
    IncrementalBeamSearch {
        // TODO: implement IncrementalBeamSearch
        beam_width: usize,
    },
    #[strum_discriminants(strum(serialize = "Ship routing"))]
    Ship {
        #[serde(default)]
        ship_mode: ShipMode,
    },
}

impl ModeConfig {
    pub(crate) fn name(&self) -> String {
        match self {
            Self::BeamSearch { beam_width, .. } => {
                format!("BS({beam_width})")
            }
            Self::IncrementalBeamSearch { beam_width } => {
                format!("IBS({beam_width})")
            }
            Self::DepthFirst => "DFS".to_string(),
            Self::IncrementalBroadening => "IB".to_string(),
            Self::BeamStack => "BSS".to_string(),
            Self::AStar { weight } => format!("AStar({weight})"),
            Self::Dijkstra => "Dijkstra".to_string(),
            Self::Ship { .. } => "Ship".to_string(),
        }
    }
}

impl std::fmt::Display for ModeConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BeamSearch {
                beam_width,
                refuel_mode,
                refuel_primary,
                boost_primary,
                range_limit,
            } => {
                    match beam_width {
                        BeamWidth::Absolute(beam_width) => {
                            write!(f, "Beam search, beam width: {beam_width}")?
                        }
                        BeamWidth::Fraction(num,den) => {
                            write!(
                                f,
                                "Beam search, beam width: {num}/{den}"
                            )?
                        }
                        BeamWidth::Infinite => {
                            write!(f, "Breadth-first search")?
                        }
                    }
                if let Some(refuel_mode) = refuel_mode {
                    write!(f, ", {refuel_mode}")?;
                    if *refuel_primary {
                        write!(f, ", only scooping at primary stars")?;
                    }
                }
                if *boost_primary {
                    write!(f, ", only supercharging at primary stars")?;
                }
                if range_limit.is_finite()
                    && *range_limit > 0.0
                    && *range_limit < 1.0
                {
                    write!(
                        f,
                        ", using at most {prc:.02}% of maximum jump range",
                        prc = range_limit * 100.0
                    )?;
                }
                Ok(())
            }
            Self::IncrementalBeamSearch { beam_width } => {
                write!(f, "Incremental Beam Search, beam width: {beam_width}")
            }
            Self::BeamStack => {
                write!(f, "Beam-stack search")
            }
            Self::DepthFirst => write!(f, "Depth-first search"),
            Self::IncrementalBroadening => {
                write!(f, "Incrementally broadening beam search")
            }
            Self::AStar { weight } => {
                write!(f, "A* shortest path, weight: {weight}")
            }
            Self::Dijkstra => write!(f, "Dijkstra shortest path"),
            Self::Ship { ship_mode: ShipMode::Fuel } => {
                write!(f, "Ship: Least fuel consumption")
            }
            Self::Ship { ship_mode: ShipMode::Jumps } => {
                write!(f, "Ship: Least number of jumps")
            }
        }
    }
}

impl std::default::Default for ModeConfig {
    fn default() -> Self {
        let ncpu = num_cpus::get();
        Self::BeamSearch {
            beam_width: BeamWidth::Absolute(1024 * ncpu),
            refuel_mode: Some(RefuelMode::WhenEmpty),
            boost_primary: false,
            refuel_primary: false,
            range_limit: f32::INFINITY,
        }
    }
}
