//! Elite: Dangerous Journal Loadout even parser
use regex::Regex;
use rustc_hash::FxHashMap;
use serde::Deserialize;

use crate::{
    common::{get_fsd_info, AstronavError, AstronavResult},
    ship::{NamedShip, Ship},
};

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct Event {
    #[serde(flatten)]
    pub event: EventData,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(tag = "event")]
pub enum EventData {
    Loadout(Loadout),
    #[serde(other)]
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct Modifier {
    label: String,
    value: f32,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct Engineering {
    modifiers: Vec<Modifier>,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct Module {
    engineering: Option<Engineering>,
    item: String,
    slot: String,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct FuelCapacity {
    main: f32,
    reserve: f32,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct Loadout {
    ship: String,
    ship_name: String,
    ship_ident: String,
    fuel_capacity: FuelCapacity,
    unladen_mass: f32,
    modules: Vec<Module>,
}

impl Engineering {
    fn to_hashmap(&self) -> FxHashMap<String, f32> {
        self.modifiers.iter().map(|v| (v.label.clone(), v.value)).collect()
    }
}

impl Loadout {
    fn get_booster(&self) -> Option<usize> {
        self.modules.iter().cloned().find_map(|m| {
            let Module { item, .. } = m;
            if item.starts_with("int_guardianfsdbooster") {
                return item
                    .chars()
                    .last()
                    .unwrap_or_else(|| unreachable!())
                    .to_digit(10)
                    .map(|v| v as usize);
            }
            None
        })
    }

    fn get_fsd(&self) -> Option<(String, Option<Engineering>)> {
        self.modules.iter().cloned().find_map(|m| {
            let Module { slot, engineering, item } = m;
            if slot == "FrameShiftDrive" {
                return Some((item, engineering));
            }
            None
        })
    }

    pub fn try_as_ship(&self) -> AstronavResult<NamedShip> {
        let fsd = self.get_fsd().ok_or_else(|| {
            AstronavError::RuntimeError("No FSD found!".to_owned())
        })?;
        let booster = self.get_booster().unwrap_or(0);
        let fsd_type = Regex::new(r"^int_hyperdrive_size(\d+)_class(\d+)$")
            .map_err(|e| format!("Failed to parse regex {e}"))?
            .captures(&fsd.0);
        let fsd_type: (usize, usize) = fsd_type
            .and_then(|m| {
                let s = m.get(1)?.as_str().to_owned().parse().ok()?;
                let c = m.get(2)?.as_str().to_owned().parse().ok()?;
                Some((c, s))
            })
            .ok_or(format!("Invalid FSD found: {}", &fsd.0))?;
        let eng = fsd.1.map(|eng| eng.to_hashmap()).unwrap_or_default();
        let mut fsd_info = get_fsd_info(fsd_type.0, fsd_type.1)?;
        let fsd_type = (
            "_EDCBA"
                .chars()
                .nth(fsd_type.0)
                .ok_or(format!("Invalid FSD found: {}", &fsd.0))?,
            fsd_type.1 as u8,
        );
        fsd_info.extend(eng);
        let max_fuel = fsd_info
            .get("MaxFuel")
            .ok_or(format!("Unknwon MaxFuelPerJump for FSD: {}", &fsd.0))?;
        let opt_mass = fsd_info
            .get("FSDOptimalMass")
            .ok_or(format!("Unknwon FSDOptimalMass for FSD: {}", &fsd.0))?;
        Ok(NamedShip {
            ship: Ship::new(
                self.unladen_mass,
                self.fuel_capacity.main,
                self.fuel_capacity.main,
                fsd_type,
                *max_fuel,
                *opt_mass,
                booster,
            )?,
            ship_type: self.ship.clone(),
            ident: self.ship_ident.clone(),
            ship_name: (!self.ship_name.is_empty())
                .then(|| self.ship_name.clone()),
        })
    }
}
