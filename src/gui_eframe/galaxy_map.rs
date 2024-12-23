use std::path::Path;

use color_eyre::eyre::Result;
use eframe::{emath::remap, epaint::ColorImage};
use fs_err::File;
use rayon::prelude::*;
use tracing::*;

use crate::common::SystemFlags;

fn scale(
    p: [f32; 3],
    p_min: [f32; 3],
    p_max: [f32; 3],
    scale: f32,
) -> [f32; 3] {
    let scale = scale - 1.0;
    let mut res = [
        remap(p[0], p_min[0]..=p_max[0], 0.0..=scale),
        remap(p[1], p_min[1]..=p_max[1], 0.0..=scale),
        remap(p[2], p_min[2]..=p_max[2], 0.0..=scale),
    ];
    res[2] = scale - res[2];
    res
}

pub fn load<P: AsRef<Path>>(path: P, size: usize) -> Result<ColorImage> {
    #[repr(packed)]
    #[derive(Clone, Copy)]
    struct Node {
        flags: SystemFlags,
        pos: [f32; 3],
    }
    let mm = unsafe { memmap2::Mmap::map(&File::open(path.as_ref())?) }?;
    let total_nodes = (mm.len() - 16) / std::mem::size_of::<Node>();
    let nodes = unsafe {
        std::slice::from_raw_parts(
            mm[16..].as_ptr().cast::<Node>(),
            total_nodes,
        )
    };
    let (p_min, p_max) = nodes
        .par_chunks(4096)
        .map(|chunk| {
            let mut acc = ([f32::MAX; 3], [f32::MIN; 3]);
            for v in chunk {
                acc = (
                    [
                        acc.0[0].min(v.pos[0]),
                        acc.0[1].min(v.pos[1]),
                        acc.0[2].min(v.pos[2]),
                    ],
                    [
                        acc.1[0].max(v.pos[0]),
                        acc.1[1].max(v.pos[1]),
                        acc.1[2].max(v.pos[2]),
                    ],
                );
            }
            acc
        })
        .collect::<Vec<_>>()
        .into_iter()
        .reduce(|acc, pos| {
            (
                [
                    acc.0[0].min(pos.0[0]),
                    acc.0[1].min(pos.0[1]),
                    acc.0[2].min(pos.0[2]),
                ],
                [
                    acc.1[0].max(pos.1[0]),
                    acc.1[1].max(pos.1[1]),
                    acc.1[2].max(pos.1[2]),
                ],
            )
        })
        .unwrap_or_default();
    let pos = nodes
        .iter()
        .map(|node| (node.flags, scale(node.pos, p_min, p_max, size as f32)));
    let mut acc = vec![vec![0f64; size]; size];
    for (flags, p) in pos {
        let px = p[0] as usize;
        let py = p[2] as usize;
        if px >= size || py >= size {
            warn!("Position: ({px},{py}) is out of range");
            continue;
        }
        acc[py][px] += flags.mult() as f64;
    }
    let max_v: f64 = acc
        .par_iter()
        .flatten()
        .max_by(|a, b| a.total_cmp(b))
        .copied()
        .unwrap_or_default()
        .log2();
    let buffer: Vec<u8> = acc
        .into_par_iter()
        .flatten()
        .flat_map(|v| {
            let alpha = if v == 0.0 { 0 } else { 255 };
            let c = colorous::CIVIDIS
                .eval_continuous(v.log2() / max_v)
                .into_array();
            [c[0], c[1], c[2], alpha]
        })
        .collect();
    Ok(ColorImage::from_rgba_unmultiplied([size, size], &buffer))
}
