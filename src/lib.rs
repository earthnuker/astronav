#![feature(substr_range)]
#![allow(clippy::cognitive_complexity, clippy::cast_precision_loss)]
#![deny(clippy::unwrap_in_result, clippy::unwrap_used, clippy::expect_used)]
#![allow(clippy::cognitive_complexity)]
#![warn(
    rust_2018_idioms,
    rust_2021_compatibility,
    arithmetic_overflow,
    nonstandard_style,
    clippy::disallowed_types,
    clippy::nursery,
    // clippy::pedantic
)]

pub(crate) mod common;
pub(crate) mod data_loader;
pub(crate) mod event;
pub(crate) mod galaxy;
pub(crate) mod journal;
pub(crate) mod route;
pub(crate) mod ship;
// mod visualizer;
pub(crate) mod sector_name;
// mod spatial_trees;

// use mimalloc::MiMalloc;

// #[global_allocator]
// static GLOBAL: MiMalloc = MiMalloc;

// #[global_allocator]
// static GLOBAL: dhat::Alloc = dhat::Alloc;

#[cfg(feature = "edmc_plugin")]
pub(crate) mod edmc_plugin;
#[cfg(feature = "gui_eframe")]
pub(crate) mod gui_eframe;
#[cfg(feature = "gui_iced")]
pub(crate) mod gui_iced;
#[cfg(feature = "pyo3")]
pub(crate) mod py_lib;
