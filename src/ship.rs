//! Ship fuel consumption and jump range calculations
use std::{
    fmt::Display,
    fs::File,
    io::{BufRead, BufReader},
    path::PathBuf,
};

use regex::Regex;
use rustc_hash::FxHashSet;
use serde::{Deserialize, Serialize};

use crate::{
    common::{get_fsd_booster_info, AstronavError, AstronavResult},
    journal::*,
};

/// Frame Shift Drive information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameShiftDrive {
    /// Rating
    pub rating_val: f32,
    /// Class
    pub class_val: f32,
    /// Optimized Mass
    pub opt_mass: f32,
    /// Max fuel per jump
    pub max_fuel: f32,
    /// Boost factor
    pub boost: f32,
    /// Guardian booster bonus range
    pub guardian_booster: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ship {
    pub base_mass: f32,
    pub fuel_mass: f32,
    pub fuel_capacity: f32,
    pub fsd: FrameShiftDrive,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamedShip {
    pub(crate) ship: Ship,
    pub ship_type: String,
    pub ident: String,
    pub ship_name: Option<String>,
}

impl PartialEq for NamedShip {
    fn eq(&self, other: &Self) -> bool {
        self.ship_type == other.ship_type
            && self.ident == other.ident
            && self.ship_name == other.ship_name
    }
}

impl Eq for NamedShip {}

impl std::hash::Hash for NamedShip {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.ship_type.hash(state);
        self.ident.hash(state);
        self.ship_name.hash(state);
    }
}

impl std::ops::DerefMut for NamedShip {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.ship
    }
}

impl std::ops::Deref for NamedShip {
    type Target = Ship;

    fn deref(&self) -> &Self::Target {
        &self.ship
    }
}

impl Display for NamedShip {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ship_name = self.ship_name.as_deref().unwrap_or("<NO NAME>");
        write!(
            f,
            "[{ident}] {name} ({ship_type}, {ship})",
            name = ship_name,
            ident = self.ident,
            ship_type = self.ship_type,
            ship = self.ship
        )
    }
}

impl NamedShip {
    pub fn get_inner(&self) -> Ship {
        self.ship.clone()
    }
}

impl From<NamedShip> for Ship {
    fn from(val: NamedShip) -> Self {
        val.ship
    }
}

impl Ship {
    fn details(&self) -> String {
        format!(
            "Ship(Mass: {:.02}+{:.02}/{:.02}t, Range: {:.02} Ly)",
            self.base_mass,
            self.fuel_mass,
            self.fuel_capacity,
            self.range()
        )
    }
}

impl Display for Ship {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Fuel: {:.02}/{:.02}t ({:.02} Ly)",
            self.fuel_mass,
            self.fuel_capacity,
            self.range()
        )
    }
}

impl Ship {
    pub fn new(
        base_mass: f32,
        fuel_mass: f32,
        fuel_capacity: f32,
        fsd_type: (char, u8),
        max_fuel: f32,
        opt_mass: f32,
        guardian_booster: usize,
    ) -> AstronavResult<Self> {
        let rating_val: f32 = match fsd_type.0 {
            'A' => 12.0,
            'B' | 'D' => 10.0,
            'C' => 8.0,
            'E' => 11.0,
            err => {
                return Err(AstronavError::RuntimeError(format!(
                    "Invalid FSD rating: {err}"
                )));
            }
        };
        if fsd_type.1 < 2 || fsd_type.1 > 8 {
            return Err(AstronavError::RuntimeError(format!(
                "Invalid FSD class: {}",
                fsd_type.1
            )));
        };
        let class_val = 0.15f32.mul_add(f32::from(fsd_type.1 - 2), 2.0);
        let ret = Self {
            fuel_capacity,
            fuel_mass,
            base_mass,
            fsd: FrameShiftDrive {
                rating_val,
                class_val,
                opt_mass,
                max_fuel,
                boost: 1.0,
                guardian_booster: get_fsd_booster_info(guardian_booster)?,
            },
        };
        Ok(ret)
    }

    pub fn new_from_json(data: &str) -> AstronavResult<NamedShip> {
        match serde_json::from_str::<Event>(data) {
            Ok(Event { event: EventData::Unknown }) => {
                Err(format!("Invalid Loadout event: {data}").into())
            }
            Ok(Event { event: EventData::Loadout(loadout) }) => {
                loadout.try_as_ship()
            }
            Err(msg) => Err(AstronavError::Other(msg.into())),
        }
    }

    pub fn new_from_loadout(loadout: &Loadout) -> AstronavResult<NamedShip> {
        loadout.try_as_ship()
    }

    pub fn new_from_journal() -> AstronavResult<Vec<NamedShip>> {
        #[cfg(not(target_os = "windows"))]
        return Err(AstronavError::Other(crate::eyre::anyhow!("Only supported on windows (for now)!")));
        let mut ret = FxHashSet::default();
        let re = Regex::new(r"^Journal\.\d{12}\.\d{2}\.log$")
            .map_err(|e| format!("Failed to parse regex: {e}"))?;
        let mut journals: Vec<PathBuf> = Vec::new();
        let mut userprofile =
            PathBuf::from(std::env::var("USERPROFILE").map_err(|e| {
                format!(
                    "Error getting journal folder location (USERPROFILE): {e}"
                )
            })?);
        userprofile.push("Saved Games");
        userprofile.push("Frontier Developments");
        userprofile.push("Elite Dangerous");
        let Ok(iter) = userprofile.read_dir() else {
            return Err("Failed to load journals".to_owned().into());
        };
        for entry in iter.flatten() {
            if re.is_match(&entry.file_name().to_string_lossy()) {
                journals.push(entry.path());
            };
        }
        journals.sort();

        for journal in &journals {
            let mut fh = BufReader::new(File::open(journal).map_err(|e| {
                format!(
                    "failed to open file {journal}: {e}",
                    journal = journal.display()
                )
            })?);
            let mut line = String::new();
            while let Ok(n) = fh.read_line(&mut line) {
                if n == 0 {
                    break;
                }
                if let Ok(Event { event: EventData::Loadout(loadout) }) =
                    serde_json::from_str::<Event>(&line)
                {
                    ret.insert(loadout.try_as_ship()?);
                };
                line.clear();
            }
        }
        if ret.is_empty() {
            return Err(AstronavError::RuntimeError(
                "No ships loaded!".to_owned(),
            ));
        }
        let mut ships = ret.into_iter().collect::<Vec<NamedShip>>();
        ships.sort_by_key(|s| format!("{s}"));
        Ok(ships)
    }

    pub fn can_jump(&self, d: f32) -> bool {
        self.fuel_cost(d) <= self.fsd.max_fuel.min(self.fuel_mass)
    }

    pub fn boost(&mut self, boost: f32) {
        self.fsd.boost = boost;
    }

    pub fn refuel(&mut self) {
        self.fuel_mass = self.fuel_capacity;
    }

    pub fn make_jump(&mut self, d: f32) -> Option<f32> {
        let cost = self.fuel_cost(d);
        if cost > self.fsd.max_fuel.min(self.fuel_mass) {
            return None;
        }
        self.fuel_mass -= cost;
        self.fsd.boost = 1.0;
        Some(cost)
    }

    pub fn jump_range(&self, fuel: f32, booster: bool) -> f32 {
        let mass = self.base_mass + fuel;
        let mut fuel = self.fsd.max_fuel.min(fuel);
        if booster {
            fuel *= self.boost_fuel_mult();
        }
        let opt_mass = self.fsd.opt_mass * self.fsd.boost;
        opt_mass
            * ((1000.0 * fuel) / self.fsd.rating_val)
                .powf(self.fsd.class_val.recip())
            / mass
    }

    pub fn max_range(&self) -> f32 {
        self.jump_range(self.fsd.max_fuel, true)
    }

    pub fn range(&self) -> f32 {
        self.jump_range(self.fuel_mass, true)
    }

    pub fn full_range(&self) -> f32 {
        self.jump_range(self.fuel_capacity, true)
    }

    fn boost_fuel_mult(&self) -> f32 {
        if self.fsd.guardian_booster == 0.0 {
            return 1.0;
        }

        let base_range = self.jump_range(self.fuel_mass, false); // current range without booster

        ((base_range + self.fsd.guardian_booster) / base_range)
            .powf(self.fsd.class_val)
    }

    pub fn fuel_cost_for_jump(
        &self,
        fuel_mass: f32,
        dist: f32,
        boost: f32,
    ) -> Option<f32> {
        if dist == 0.0 {
            return Some(0.0);
        }
        let base_cost =
            dist * ((self.base_mass + fuel_mass) / (self.fsd.opt_mass * boost));
        let fuel_cost =
            (self.fsd.rating_val * 0.001 * base_cost.powf(self.fsd.class_val))
                / self.boost_fuel_mult();
        if fuel_cost > self.fsd.max_fuel || fuel_cost > fuel_mass {
            return None;
        };
        Some(fuel_cost)
    }

    pub fn fuel_cost(&self, d: f32) -> f32 {
        if d == 0.0 {
            return 0.0;
        }
        let mass = self.base_mass + self.fuel_mass;
        let opt_mass = self.fsd.opt_mass * self.fsd.boost;
        let base_cost = (d * mass) / opt_mass;
        (self.fsd.rating_val * 0.001 * base_cost.powf(self.fsd.class_val))
            / self.boost_fuel_mult()
    }
}
