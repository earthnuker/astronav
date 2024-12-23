use std::fmt::Display;

use color_eyre::eyre::Result;
use human_repr::{HumanCount, HumanDuration, HumanThroughput};
use serde::{Deserialize, Serialize};

use crate::route::ModeConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteState {
    pub mode: ModeConfig,
    pub workers: usize,
    pub system: String,
    pub from: String,
    pub to: String,
    pub depth: usize,
    pub queue_size: usize,
    pub d_rem: f32,
    pub d_total: f32,
    pub prc_done: f32,
    pub n_seen: usize,
    pub prc_seen: f32,
    pub rate: f64,
    pub refuels: Option<usize>,
    pub msg: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessState {
    pub prc_done: f64,
    pub bytes_rate: f64,
    pub sys_rate: f64,
    pub t_rem: f64,
    pub num_systems: u64,
    pub num_errors: u64,
    pub index_size: u64,
    pub data_size: u64,
    pub names_size: u64,
    pub bytes_read: u64,
    pub uncomp_rate: f64,
    pub msg: Option<String>,
    pub file_pos: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "event")]
pub enum Event {
    SearchState(RouteState),
    ProcessState(ProcessState),
    Message(String),
}

impl Display for RouteState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self {
            mode,
            workers,
            system: _,
            from,
            to,
            depth,
            queue_size,
            d_rem,
            d_total,
            prc_done,
            n_seen,
            prc_seen,
            rate,
            msg,
            refuels,
        } = self;
        let mode = mode.name();
        let d_rem = d_rem.human_count("Ly");
        let d_total = d_total.human_count("Ly");
        let queued = queue_size.human_count_bare();
        let seen = n_seen.human_count_bare();
        let rate = rate.human_throughput("systems");
        let mode =
            if *workers > 1 { format!("{mode} {workers}x") } else { mode };
        write!(
            f,
            "[{prc_done:.02}% | {mode} | {from} -> {to}] Depth: {depth} | Remaining distance: {d_rem:.02} / {d_total:.02} |"
        )?;
        if let Some(refuels) = refuels {
            let refuels = refuels.human_count_bare();
            write!(f, " Refuels: {refuels} |")?;
        }
        write!(
            f,
            " Queue: {queued} | Visited: {seen} ({prc_seen:.02}%) | Rate: {rate}"
        )?;
        if let Some(msg) = msg.as_ref() {
            write!(f, " | {msg}")?;
        };
        Ok(())
    }
}

impl Display for ProcessState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self {
            prc_done,
            bytes_rate,
            sys_rate,
            t_rem,
            num_systems,
            num_errors,
            msg,
            index_size,
            data_size,
            names_size,
            uncomp_rate,
            bytes_read,
            file_pos,
        } = self;

        let sys_rate = sys_rate.human_throughput_bare();
        let bytes_rate = bytes_rate.human_throughput_bytes();
        let t_rem = t_rem.human_duration();
        let num_systems = num_systems.human_count_bare();
        let total_size =
            (index_size + data_size + names_size).human_count_bytes();
        let index_size = index_size.human_count_bytes();
        let data_size = data_size.human_count_bytes();
        let names_size = names_size.human_count_bytes();
        let bytes_read = bytes_read.human_count_bytes();
        let file_pos = file_pos.human_count_bytes();
        let uncomp_rate = uncomp_rate.human_throughput_bytes();
        write!(
            f,
            "[{prc_done:.2}%] | {num_systems} system(s) written ({file_pos} read @ {bytes_rate}, {bytes_read} decompressed @ {uncomp_rate}, {sys_rate}, Index: {index_size}, Data: {data_size}, Names: {names_size}, Total: {total_size}), {num_errors} error(s), ETA: {t_rem}"
        )?;
        if let Some(msg) = msg.as_ref() {
            write!(f, " | {msg}")?;
        }
        Ok(())
    }
}

impl Display for Event {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SearchState(state) => {
                write!(f, "{state}")?;
            }
            Self::ProcessState(state) => {
                write!(f, "{state}")?;
            }
            Self::Message(msg) => {
                write!(f, "{msg}")?;
            }
        }
        Ok(())
    }
}

pub type Callback =
    Box<dyn Fn(&crate::route::Router, &Event) -> Result<()> + Send>;
