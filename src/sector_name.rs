#![allow(dead_code)]
use std::{collections::BTreeMap, default, str::FromStr};

use color_eyre::eyre::{anyhow, bail, Result};
use itertools::Itertools;
use lazy_static::lazy_static;
use regex::{Match, Regex};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use crate::common::System;

fn split_list(s: &str) -> Vec<String> {
    s.split(',').map(|s| s.trim().to_owned().to_lowercase()).collect_vec()
}

fn jenkins_hash(mut n: u32) -> u32 {
    for (add_shift, xor_shift) in &[(12, 22), (4, 9), (10, 2), (7, 12)] {
        n = n.wrapping_add(n << add_shift);
        n ^= n >> xor_shift;
    }
    n
}

lazy_static! {
    static ref PREFIXES: Vec<String> = split_list("Th, Eo, Oo, Eu, Tr, Sly, Dry, Ou, Tz, Phl, Ae, Sch, Hyp, Syst, Ai, Kyl, Phr, Eae, Ph, Fl, Ao, Scr, Shr, Fly, Pl, Fr, Au, Pry, Pr, Hyph, Py, Chr, Phyl, Tyr, Bl, Cry, Gl, Br, Gr, By, Aae, Myc, Gyr, Ly, Myl, Lych, Myn, Ch, Myr, Cl, Rh, Wh, Pyr, Cr, Syn, Str, Syr, Cy, Wr, Hy, My, Sty, Sc, Sph, Spl, A, Sh, B, C, D, Sk, Io, Dr, E, Sl, F, Sm, G, H, I, Sp, J, Sq, K, L, Pyth, M, St, N, O, Ny, Lyr, P, Sw, Thr, Lys, Q, R, S, T, Ea, U, V, W, Schr, X, Ee, Y, Z, Ei, Oe");
    static ref INFIXES_1: Vec<String> = split_list("o, ai, a, oi, ea, ie, u, e, ee, oo, ue, i, oa, au, ae, oe");
    static ref INFIXES_2: Vec<String> = split_list("ll, ss, b, c, d, f, dg, g, ng, h, j, k, l, m, n, mb, p, q, gn, th, r, s, t, ch, tch, v, w, wh, ck, x, y, z, ph, sh, ct, wr");
    static ref SUFFIXES_1: Vec<String> = split_list("oe, io, oea, oi, aa, ua, eia, ae, ooe, oo, a, ue, ai, e, iae, oae, ou, uae, i, ao, au, o, eae, u, aea, ia, ie, eou, aei, ea, uia, oa, aae, eau, ee");
    static ref SUFFIXES_2: Vec<String> = split_list("b, scs, wsy, c, d, vsky, f, sms, dst, g, rb, h, nts, ch, rd, rld, k, lls, ck, rgh, l, rg, m, n, hm, p, hn, rk, q, rl, r, rm, s, cs, wyg, rn, ct, t, hs, rbs, rp, tts, v, wn, ms, w, rr, mt, x, rs, cy, y, rt, z, ws, lch, my, ry, nks, nd, sc, ng, sh, nk, sk, nn, ds, sm, sp, ns, nt, dy, ss, st, rrs, xt, nz, sy, xy, rsch, rphs, sts, sys, sty, th, tl, tls, rds, nch, rns, ts, wls, rnt, tt, rdy, rst, pps, tz, tch, sks, ppy, ff, sps, kh, sky, ph, lts, wnst, rth, ths, fs, pp, ft, ks, pr, ps, pt, fy, rts, ky, rshch, mly, py, bb, nds, wry, zz, nns, ld, lf, gh, lks, sly, lk, ll, rph, ln, bs, rsts, gs, ls, vvy, lt, rks, qs, rps, gy, wns, lz, nth, phs");
    static ref SEQUENCES: Vec<Vec<&'static Vec<String>>> = {
        let mut seqs = vec![];
        for infix_1 in &[&*INFIXES_1,&*INFIXES_2] {
            for suff_1 in &[&*SUFFIXES_1,&*SUFFIXES_2] {
                seqs.push(vec![&*PREFIXES,infix_1,suff_1]);
                for infix_2 in &[&*INFIXES_1,&*INFIXES_2] {
                    seqs.push(vec![&*PREFIXES,infix_1,infix_2,suff_1]);
                }
            }
        }
        for suff_1 in &[&*SUFFIXES_1,&*SUFFIXES_2] {
            for suff_2 in &[&*SUFFIXES_1,&*SUFFIXES_2] {
                seqs.push(vec![&*PREFIXES,suff_1,&*PREFIXES,suff_2]);
            }
        }
        dbg!(PREFIXES.len());
        dbg!(INFIXES_1.len());
        dbg!(INFIXES_2.len());
        dbg!(SUFFIXES_1.len());
        dbg!(SUFFIXES_2.len());
        dbg!(seqs.len());
        seqs
    };
    static ref SECTOR_REGEX: Regex = Regex::new(r"^(?P<sector>[\w\s'.()/-]+) (?P<l1>[A-Za-z])(?P<l2>[A-Za-z])-(?P<l3>[A-Za-z]) (?P<mcode>[A-Za-z])(?:(?P<n1>\d+)-)?(?P<n2>\d+)$").unwrap_or_else(|_| unreachable!());
}

fn match_sequence(
    name: &str,
    chunks: &[&Vec<String>],
    seq: &mut Vec<(usize, String)>,
) -> Option<Vec<usize>> {
    if name.is_empty() && chunks.is_empty() {
        return Some(seq.iter().map(|(n, _)| *n).collect());
    }
    if chunks.is_empty() {
        return None;
    }
    for (n, prefix) in chunks[0].iter().enumerate() {
        if let Some(name) = name.strip_prefix(prefix) {
            seq.push((n, prefix.to_owned()));
            if let Some(ret) = match_sequence(name, &chunks[1..], seq) {
                return Some(ret);
            };
            seq.pop();
        }
    }
    None
}

fn parse_sector(sector_name: String) -> Result<Sector> {
    let name = sector_name
        .to_ascii_lowercase()
        .replace(|c: char| c.is_ascii_whitespace(), "");
    for (idx, seq) in SEQUENCES.iter().enumerate() {
        if let Some(res) = match_sequence(&name, seq, &mut Vec::new()) {
            return Ok(Sector { name: sector_name, idx, pos: res });
        };
    }
    bail!("Failed to parse sector name: {name:?}!");
}

#[derive(Debug)]
struct Sector {
    name: String,
    pos: Vec<usize>,
    idx: usize,
}

#[derive(Debug)]
struct SystemInfo {
    name: String,
    sector: Sector,
    l: [usize; 3],
    mcode: char,
    n: [usize; 2],
}

fn to_base(mut n: usize, d: usize) -> Vec<usize> {
    let mut res = vec![];
    while n != 0 {
        res.push(n % d);
        n /= d;
    }
    res.reverse();
    res
}

fn from_base(d: &[usize], b: usize) -> usize {
    let mut res: usize = 0;
    for n in d {
        res = (res * b) + n;
    }
    res
}

impl SystemInfo {
    const BASE_SECTOR_INDEX: [f32; 3] = [39., 32., 18.];
    const ORIGIN: [f32; 3] = [-49985., -40985., -24105.];
    const SECTOR_SIZE: usize = 1280;

    fn get_sector_position(&self) -> Result<[usize; 3]> {
        let mut sector_offset =
            vec![self.n[0], self.l[0], self.l[1], self.l[2]];
        sector_offset.reverse();
        let index = from_base(&sector_offset, 26);
        let mut sector_pos = to_base(index, 128);
        if sector_pos.len() > 3 {
            bail!("Sector position out of range for {index}");
        }
        sector_pos.resize(3, 0);
        dbg!(sector_offset, &sector_pos);
        sector_pos.try_into().map_err(|_| anyhow!("Invalid sector position!"))
    }

    fn get_relative_position(&self) -> Result<[usize; 3]> {
        let mut out_of_range = false;
        let cube_width: usize =
            Self::SECTOR_SIZE >> (b'h' - (self.mcode as u8));
        let half_cube = cube_width >> 1;
        let mut relpos = [half_cube; 3];
        for (i, n) in relpos.into_iter().rev().enumerate() {
            let v = n * cube_width;
            if v > Self::SECTOR_SIZE + cube_width {
                out_of_range = true;
            }
            relpos[i] = v;
        }
        dbg!(cube_width, relpos, out_of_range);
        if out_of_range {
            bail!("Potision out of range for {}", self.name);
        }
        Ok(relpos)
    }

    fn get_pos(&self) -> Result<([f32; 3], f32)> {
        let cube_width: usize =
            Self::SECTOR_SIZE >> (b'h' - (self.mcode as u8));
        let rel_pos = self.get_relative_position()?;
        let sec_pos = self.get_sector_position()?;
        dbg!(sec_pos, rel_pos);
        todo!();
        // let ret = [
        //     Self::ORIGIN[0]+relpos[0],
        //     Self::ORIGIN[1]+relpos[1],
        //     Self::ORIGIN[2]+relpos[2],
        // ];
        // Ok((ret, (cube_width>>1) as f32))
    }
}

impl FromStr for SystemInfo {
    type Err = color_eyre::eyre::Report;

    fn from_str(orig_name: &str) -> Result<Self> {
        let name = orig_name.to_lowercase();
        let mut res: FxHashMap<&str, Option<&str>> = FxHashMap::default();
        if let Some(c) = SECTOR_REGEX.captures(&name) {
            for (name, m) in SECTOR_REGEX.capture_names().zip(c.iter()) {
                let m = m.map(|m| m.as_str());
                if let Some(name) = name {
                    res.insert(name, m);
                }
            }
        }
        if res.is_empty() {
            bail!("Failed to parse system name");
        }
        Ok(Self {
            name: orig_name.to_owned(),
            sector: parse_sector(
                res["sector"]
                    .ok_or_else(|| anyhow!("Sector not found!"))?
                    .to_owned(),
            )?,
            l: [
                res["l1"]
                    .and_then(|c| c.to_uppercase().chars().next())
                    .ok_or_else(|| anyhow!("L1 not found!"))?
                    as usize
                    - 0x41,
                res["l2"]
                    .and_then(|c| c.to_uppercase().chars().next())
                    .ok_or_else(|| anyhow!("L2 not found!"))?
                    as usize
                    - 0x41,
                res["l3"]
                    .and_then(|c| c.to_uppercase().chars().next())
                    .ok_or_else(|| anyhow!("L3 not found!"))?
                    as usize
                    - 0x41,
            ],
            mcode: res["mcode"]
                .and_then(|c| c.chars().next())
                .ok_or_else(|| anyhow!("MCode not found!"))?,
            n: [
                res["n1"].map(|n| n.parse()).transpose()?.unwrap_or(0),
                res["n2"].map(|n| n.parse()).transpose()?.unwrap_or(0),
            ],
        })
    }
}

fn parse_name(name: &str) -> Result<SystemInfo> {
    // http://disc.thargoid.space/Sector_Naming
    // https://bitbucket.org/Esvandiary/edts/src/develop/edtslib/pgnames.py
    // https://bitbucket.org/Esvandiary/edts/src/develop/edtslib/pgdata.py
    // https://bitbucket.org/Esvandiary/edts/src/develop/test/unit/test_pgnames.py
    // https://edts.thargoid.space/
    // dbg!(name);
    let sys_info: SystemInfo = name.parse()?;
    // dbg!(sys_info.get_pos());
    Ok(sys_info)
}

#[derive(Debug, Serialize, Deserialize, Default)]
struct Node {
    value: Option<u32>,
    children: BTreeMap<u8, Self>,
}

impl Node {
    fn new() -> Self {
        Default::default()
    }

    fn insert(&mut self, key: &[u8], value: u32) -> Option<u32> {
        if key.is_empty() {
            return self.value.replace(value);
        }
        self.children.entry(key[0]).or_default().insert(&key[1..], value)
    }
}

#[cfg(test)]
mod test {
    use std::{
        collections::BTreeMap,
        io::{BufWriter, Write},
        iter,
        sync::{atomic::AtomicU64, Mutex},
    };

    use super::*;
    #[test]
    fn test_pgnames() -> Result<()> {
        use std::{fs::File, sync::atomic::Ordering};

        use byteorder::*;
        use memmap2::Mmap;
        use rayon::prelude::*;
        let key_distr: Mutex<BTreeMap<u8, usize>> = Default::default();
        let total_seqs = SEQUENCES.len();
        if let Ok(names_path) = std::env::var("STARS_PATH") {
            let (tx, rx) = crossbeam_channel::bounded::<([u8; 12], u32)>(4096);
            let mm_names = unsafe { Mmap::map(&File::open(names_path)?) }?;
            let job_handle = std::thread::spawn(move || {
                mm_names.split(|&c| c == b'\n')
                .filter(|c| c.first() != Some(&b'#'))
                .enumerate()
                .par_bridge()
                .for_each(|(n,name)| {
                    let Ok(name) = std::str::from_utf8(name) else {
                        return;
                    };
                    let Ok(res) = parse_name(name) else {
                        return
                    };
                    let name = res
                        .name
                        .chars()
                        .zip(
                            res.sector
                                .name
                                .chars()
                                .chain(iter::repeat('\0')),
                        )
                        .filter_map(|(a, b)| {
                            let a_l = a.to_lowercase().to_string();
                            let b_l = b.to_lowercase().to_string();
                            (a_l!=b_l).then_some(a)
                        })
                        .filter(|c| c.is_alphabetic())
                        .map(|c| c as u8)
                        .collect::<Vec<u8>>();
                    let mut tree_key = vec![];
                    tree_key.push(res.sector.idx.try_into().unwrap());
                    tree_key.extend(res.sector.pos.into_iter().map(|c| u8::try_from(c).unwrap()));
                    tree_key.extend(name);
                    tree_key.extend(u8::try_from(res.n[0]).unwrap().to_be_bytes());
                    tree_key.extend(u16::try_from(res.n[1]).unwrap().to_be_bytes());
                    tree_key.resize(12, 0);
                    let tree_key: [u8;12] = tree_key.try_into().unwrap();
                    let mut key_distr=key_distr.lock().unwrap();
                    let entry = key_distr.entry(tree_key[0]).or_default();
                    *entry+=1;
                    if n%100_000==0 {
                        let key_len = tree_key.len();
                        println!("[{n}] {tree_key:?} {key_len} {key_distr:?}/{total_seqs}");
                    }
                    let value = (tree_key,n.try_into().unwrap());
                    tx.send(value).unwrap();
                })
            });
            let mut fh = BufWriter::new(
                File::create(r"C:\Users\Earthnuker\AppData\Local\astronav\data\pg_names.bin")
                    .unwrap(),
            );
            let buffer: BTreeMap<[u8; 12], u32> = rx.into_iter().collect();
            for (key, value) in buffer.into_iter() {
                fh.write_all(&key)?;
                fh.write_u32::<LittleEndian>(value)?;
            }
            job_handle.join().unwrap();
        }
        // dbg!(parse_sector("Dryau Aowsy".to_owned()));
        // dbg!(parse_name("Blae Drye FU-K a76-1"))?; // ( -1773.31, -58.72,
        // 4867.72) dbg!(parse_name("Synookio DA-Q b19-7"))?; //
        // (1130.44, 78.78, 12157.63)
        Ok(())
    }
}
