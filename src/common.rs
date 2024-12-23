//! # Common utlility functions
use std::{
    cmp::{Ordering, Reverse},
    collections::BinaryHeap,
    fmt::Display,
    hash::{Hash, Hasher},
    io::{BufReader, Read},
    num::{NonZeroUsize, ParseIntError},
    ops::{Deref, DerefMut},
    path::{Path, PathBuf},
    str::FromStr,
    time::Instant,
};

use byteorder::{LittleEndian, ReadBytesExt};
use color_eyre::{
    eyre,
    eyre::{anyhow, Result},
};
use fs_err::File;
use human_repr::HumanDuration;
use num_format::ToFormattedString;
use parse_display::{Display, FromStr};
#[cfg(feature = "pyo3")]
use pyo3::PyErr;
use rustc_hash::FxHashMap;
use serde::{ser::SerializeMap, Deserialize, Serialize};
use sif_kdtree::Object;
use thiserror::Error;
use tracing::error;
use tracing_subscriber::fmt::{format::Writer, time::FormatTime};

use crate::route::{Router,RouterTree};

pub trait FormatNum {
    fn format_num(&self) -> String;
}

pub trait FormatFloat {
    fn format_float(&self) -> String;
}

impl<N: ToFormattedString> FormatNum for N {
    fn format_num(&self) -> String {
        use num_format::{Locale, SystemLocale};
        let locale = SystemLocale::default()
            .and_then(|l| l.name().parse())
            .unwrap_or(Locale::en);
        self.to_formatted_string(&locale)
    }
}

impl<N: Copy + Into<f64>> FormatFloat for N {
    fn format_float(&self) -> String {
        use format_num::NumberFormat;
        let num = NumberFormat::new();
        let val: f64 = (*self).into();
        num.format(",.2", val)
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct RelativeTime {
    epoch: Instant,
}

impl Default for RelativeTime {
    fn default() -> Self {
        Self { epoch: Instant::now() }
    }
}

impl FormatTime for RelativeTime {
    fn format_time(&self, w: &mut Writer<'_>) -> core::fmt::Result {
        let duration = format!("{}", self.epoch.elapsed().human_duration());
        write!(w, "{}", duration)
    }
}

#[allow(clippy::inline_always)]
/// heuristic used for Beam and A* search
pub fn heuristic(
    range: f32,
    node: &TreeNode,
    goal: &TreeNode,
    primary_only: bool,
) -> f32 {
    const FAST: bool = true;
    if std::ptr::addr_eq(node, goal) {
        // ensure goal has lowest heuristic
        return f32::NEG_INFINITY;
    }
    let mult = if primary_only {
        node.primary_mult()
    } else {
        node.mult()
    };

    if FAST {
        // Fast and decent heuristic
        range.mul_add(-mult, dist(node.pos(), goal.pos())).max(0.0)
    } else {
        // slightly better heuristic but significantly slower due to the division
        dist(node.pos(), goal.pos())/(range * mult)
    }
}

pub fn sort_by_heuristic(
    nodes: &mut [&TreeNode],
    goal: &TreeNode,
    range: f32,
    primary_only: bool,
) {
    nodes.sort_by_key(|node| F32(heuristic(range, node, goal, primary_only)))
}

/// Min-heap priority queue using f32 as priority
pub struct MinFHeap<T: Ord>(BinaryHeap<(Reverse<F32>, T)>);

impl<T: Ord> MinFHeap<T> {
    /// Create new, empty priority queue
    pub const fn new() -> Self {
        Self(BinaryHeap::new())
    }

    /// push value `item` with priority `w` into queue
    pub fn push(&mut self, w: f32, item: T) {
        self.0.push((Reverse(F32(w)), item));
    }

    /// Remove and return smallest item and priority
    pub fn pop(&mut self) -> Option<(f32, T)> {
        self.0.pop().map(|(Reverse(F32(w)), item)| (w, item))
    }

    pub fn clear(&mut self) {
        self.0.clear();
    }
}

impl<T: Ord> Default for MinFHeap<T> {
    fn default() -> Self {
        Self(BinaryHeap::new())
    }
}

impl<T: Ord> Deref for MinFHeap<T> {
    type Target = BinaryHeap<(Reverse<F32>, T)>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: Ord> DerefMut for MinFHeap<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T: Ord> Extend<(f32, T)> for MinFHeap<T> {
    fn extend<I: IntoIterator<Item = (f32, T)>>(&mut self, iter: I) {
        self.0.extend(iter.into_iter().map(|(w, v)| (Reverse(F32(w)), v)))
    }
}

// pub struct MinHeap<K: Ord, V: Ord>(BinaryHeap<(Reverse<K>, V)>);

// impl<K: Ord, V: Ord> MinHeap<K, V> {
//     /// Create new, empty priority queue
//     pub fn new() -> Self {
//         Self(BinaryHeap::new())
//     }

//     /// push value `item` with priority `w` into queue
//     pub fn push(&mut self, w: K, item: V) {
//         self.0.push((Reverse(w), item));
//     }

//     /// Remove and return smallest item and priority
//     pub fn pop(&mut self) -> Option<(K, V)> {
//         self.0.pop().map(|(Reverse(k), item)| (k, item))
//     }
// }

// impl<K: Ord, V: Ord> Default for MinHeap<K, V> {
//     fn default() -> Self {
//         Self(BinaryHeap::new())
//     }
// }

// impl<K: Ord, V: Ord> Deref for MinHeap<K, V> {
//     type Target = BinaryHeap<(Reverse<K>, V)>;
//     fn deref(&self) -> &Self::Target {
//         &self.0
//     }
// }

// impl<K: Ord, V: Ord> DerefMut for MinHeap<K, V> {
//     fn deref_mut(&mut self) -> &mut Self::Target {
//         &mut self.0
//     }
// }

// struct NodeHeap<'a> {
//     range: f32,
//     goal: &'a TreeNode,
//     heap: MinFHeap<&'a TreeNode>,
// }

// impl<'a> NodeHeap<'a> {
//     fn new(range: f32, goal: &'a TreeNode) -> Self {
//         Self { range, goal, heap: MinFHeap::default() }
//     }

//     fn push(&mut self, node: &'a TreeNode) {
//         let h = heuristic(self.range, node, self.goal, false);
//         self.heap.push(h, node);
//     }
// }

/// ED LRR error type
#[derive(Error, Debug)]
pub enum AstronavError {
    #[error("failed to compute route from {from} to {to}: {reason}")]
    RouteError { from: System, to: System, reason: Box<Self> },

    #[error("system matching {0} not found")]
    SystemNotFoundError(SysEntry),

    #[error("{0}")]
    RuntimeError(String),

    #[error(transparent)]
    IOError(#[from] std::io::Error),

    #[error(transparent)]
    BincodeError(#[from] Box<bincode::ErrorKind>),

    #[cfg(feature = "pyo3")]
    #[error(transparent)]
    PyError(#[from] pyo3::PyErr),

    #[error("{0:#}")]
    Other(#[from] eyre::Report),
}

#[cfg(feature = "pyo3")]
pub mod py_exceptions {
    use pyo3::create_exception;
    pub use pyo3::exceptions::*;

    create_exception!(astronav, RouteError, PyException);
    create_exception!(astronav, SystemNotFoundError, PyException);
    create_exception!(astronav, AstronavException, PyException);
    create_exception!(astronav, FileFormatError, PyException);
    create_exception!(astronav, EvalError, PyException);
}
impl FromStr for AstronavError {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self::RuntimeError(s.to_owned()))
    }
}

impl std::convert::From<String> for AstronavError {
    fn from(s: String) -> Self {
        Self::RuntimeError(s)
    }
}

#[cfg(feature = "pyo3")]
impl std::convert::From<AstronavError> for PyErr {
    fn from(err: AstronavError) -> Self {
        match err {
            #[cfg(feature = "pyo3")]
            AstronavError::PyError(e) => e,
            AstronavError::BincodeError(..) => {
                py_exceptions::FileFormatError::new_err(err.to_string())
            }
            AstronavError::RouteError { .. } => {
                py_exceptions::RouteError::new_err(err.to_string())
            }
            AstronavError::RuntimeError(e) => {
                py_exceptions::PyRuntimeError::new_err(e)
            }
            AstronavError::SystemNotFoundError(..) => {
                py_exceptions::SystemNotFoundError::new_err(err.to_string())
            }
            AstronavError::IOError(e) => {
                py_exceptions::PyIOError::new_err(e.to_string())
            }
            AstronavError::Other(e) => {
                py_exceptions::AstronavException::new_err(e.to_string())
            }
        }
    }
}

pub type AstronavResult<T> = Result<T, AstronavError>;

/// f32 compare wrapper
pub fn fcmp(a: f32, b: f32) -> Ordering {
    match (a, b) {
        (x, y) if x.is_nan() && y.is_nan() => Ordering::Equal,
        (x, _) if x.is_nan() => Ordering::Greater,
        (_, y) if y.is_nan() => Ordering::Less,
        (..) => a.partial_cmp(&b).unwrap_or_else(|| unreachable!()),
    }
}

/// f32 warpper type implementing `Eq` and `Ord`
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct F32(pub f32);

impl Display for F32 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl PartialEq for F32 {
    fn eq(&self, other: &Self) -> bool {
        fcmp(self.0, other.0) == std::cmp::Ordering::Equal
    }
}

impl Eq for F32 {}

impl PartialOrd for F32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for F32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        fcmp(self.0, other.0)
    }
}

impl Deref for F32 {
    type Target = f32;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for F32 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Returns additional jump range (in Ly) granted by specified class of Guardian
/// FSD Booster
pub fn get_fsd_booster_info(class: usize) -> Result<f32, String> {
    // Data from https://elite-dangerous.fandom.com/wiki/Guardian_Frame_Shift_Drive_Booster
    let ret = match class {
        0 => 0.0,
        1 => 4.0,
        2 => 6.0,
        3 => 7.75,
        4 => 9.25,
        5 => 10.5,
        _ => return Err(format!("Invalid Guardian booster class: {class}")),
    };
    Ok(ret)
}

/// Returns optimal mass and maximum fuel per jump for the given FSD rating and
/// class as a hash map
pub fn get_fsd_info(
    rating: usize,
    class: usize,
) -> Result<FxHashMap<String, f32>, String> {
    let mut ret = FxHashMap::default();
    // Data from https://elite-dangerous.fandom.com/wiki/Frame_Shift_Drive#Specifications
    let (opt_mass, max_fuel) = match (class, rating) {
        (2, 1) => (48.0, 0.6),
        (2, 2) => (54.0, 0.6),
        (2, 3) => (60.0, 0.6),
        (2, 4) => (75.0, 0.8),
        (2, 5) => (90.0, 0.9),

        (3, 1) => (80.0, 1.2),
        (3, 2) => (90.0, 1.2),
        (3, 3) => (100.0, 1.2),
        (3, 4) => (125.0, 1.5),
        (3, 5) => (150.0, 1.8),

        (4, 1) => (280.0, 2.0),
        (4, 2) => (315.0, 2.0),
        (4, 3) => (350.0, 2.0),
        (4, 4) => (438.0, 2.5),
        (4, 5) => (525.0, 3.0),

        (5, 1) => (560.0, 3.3),
        (5, 2) => (630.0, 3.3),
        (5, 3) => (700.0, 3.3),
        (5, 4) => (875.0, 4.1),
        (5, 5) => (1050.0, 5.0),

        (6, 1) => (960.0, 5.3),
        (6, 2) => (1080.0, 5.3),
        (6, 3) => (1200.0, 5.3),
        (6, 4) => (1500.0, 6.6),
        (6, 5) => (1800.0, 8.0),

        (7, 1) => (1440.0, 8.5),
        (7, 2) => (1620.0, 8.5),
        (7, 3) => (1800.0, 8.5),
        (7, 4) => (2250.0, 10.6),
        (7, 5) => (2700.0, 12.8),
        (r, c) => {
            return Err(format!("Invalid FSD Type: Rating: {r}, Class: {c}"));
        }
    };
    ret.insert("FSDOptimalMass".to_owned(), opt_mass);
    ret.insert("MaxFuel".to_owned(), max_fuel);
    Ok(ret)
}

#[derive(
    Debug, Copy, Clone, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord,
)]
#[serde(untagged)]
pub enum BeamWidth {
    Absolute(usize),
    Fraction(usize, NonZeroUsize),
    #[serde(deserialize_with = "bw_infinite")]
    Infinite,
}

impl Default for BeamWidth {
    fn default() -> Self {
        Self::Absolute(1024 * rayon::current_num_threads())
    }
}

fn bw_infinite<'de, D>(deserializer: D) -> Result<(), D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(rename_all = "snake_case")]
    enum Helper {
        Infinite,
        Inf,
    }
    Helper::deserialize(deserializer)?;
    Ok(())
}

impl FromStr for BeamWidth {
    type Err = AstronavError;

    fn from_str(bw: &str) -> Result<Self, Self::Err> {
        let s = bw.to_lowercase();
        match s.as_str() {
            "inf" | "infinite" => Ok(Self::Infinite),
            other => {
                if let Ok(val) = other.parse() {
                    return Ok(Self::Absolute(val));
                }
                if let &[num, denom] =
                    s.split('/').collect::<Vec<&str>>().as_slice()
                {
                    let num = num.parse().map_err(|e: ParseIntError| {
                        AstronavError::Other(e.into())
                    })?;
                    let denom = denom.parse().map_err(|e: ParseIntError| {
                        AstronavError::Other(e.into())
                    })?;
                    return Ok(Self::Fraction(num, denom));
                }
                Err(AstronavError::Other(anyhow!(
                    "Failed to parse beam width: {bw}"
                )))
            }
        }
    }
}

impl std::fmt::Display for BeamWidth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Absolute(n) => write!(f, "{n}"),
            Self::Fraction(n, d) => write!(f, "{n}/{d}"),
            Self::Infinite => write!(f, "Infinite"),
        }?;
        Ok(())
    }
}

impl BeamWidth {
    pub const fn needs_sort(&self) -> bool {
        match self {
            Self::Absolute(_) | Self::Fraction(_, _) => true,
            Self::Infinite => false,
        }
    }

    pub fn compute(&self, nodes: usize) -> usize {
        match self {
            Self::Fraction(n, d) => {
                let d = d.get();
                if *n > d {
                    return nodes;
                }
                (n * nodes) / d
            }
            Self::Absolute(n) => *n,
            Self::Infinite => nodes,
        }
    }
}
/// Represents an uresolved system to be searched for by name, id or position

#[derive(Debug, Clone, Display, FromStr, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SysEntry {
    #[display(":{0}")]
    ID(u32),
    #[display("#{0}")]
    ID64(u64),
    #[display("{0}/{1}/{2}")]
    Pos(f32, f32, f32),
    #[display("{0}")]
    Name(String),
}

pub fn is_older(file_a: &Path, file_b: &Path) -> bool {
    if !file_b.exists() {
        return false;
    }
    let mod_a = std::fs::metadata(file_a).and_then(|m| m.modified());
    let mod_b = std::fs::metadata(file_b).and_then(|m| m.modified());
    if let (Ok(mod_a), Ok(mod_b)) = (mod_a, mod_b) {
        return mod_a > mod_b;
    }
    false
}

#[allow(clippy::inline_always)]
pub fn line_dist(node: &[f32; 3], start: &[f32; 3], end: &[f32; 3]) -> f32 {
    let c = dist(start, end);
    let a = dist(node, end);
    let b = dist(start, node);
    ((a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c)).sqrt() / (c * 2.0)
}

#[allow(clippy::inline_always)]
pub fn dist2(p1: &[f32; 3], p2: &[f32; 3]) -> f32 {
    p1.iter().zip(p2.iter()).map(|(a, b)| (a - b).powi(2)).sum()
}

#[allow(clippy::inline_always)]
pub fn dist(p1: &[f32; 3], p2: &[f32; 3]) -> f32 {
    dist2(p1, p2).sqrt()
}

#[repr(u8)]
#[derive(Debug, Serialize, PartialEq, Eq, PartialOrd, Ord, Copy, Clone)]
#[serde(rename_all = "snake_case")]
pub enum StarKind {
    Regular = 0b00,   // Unscoopable, Black Hole, None, etc
    Scoopable = 0b01, // KGBFOAM
    WhiteDwarf = 0b10,
    Neutron = 0b11,
}

impl StarKind {
    const NUM_BITS: usize = 2;

    const fn to_bits(self) -> u8 {
        self as u8
    }

    const fn tag(self) -> char {
        match self {
            Self::Regular => 'r',
            Self::Scoopable => 's',
            Self::WhiteDwarf => 'w',
            Self::Neutron => 'n',
        }
    }

    const fn mult(self) -> f32 {
        match self {
            Self::WhiteDwarf => 1.5,
            Self::Neutron => 4.0,
            _ => 1.0,
        }
    }
}

// [00_wr_kk_pp], p=primary_kind, k=kind, r=refuel, w=waypoint
#[repr(transparent)]
#[derive(
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    bytemuck::Pod,
    bytemuck::Zeroable,
)]
pub struct SystemFlags(u8);

impl std::fmt::Debug for SystemFlags {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SystemFlags")
            .field("value", &self.0)
            .field("primary_kind", &self.primary_kind())
            .field("kind", &self.kind())
            .field("waypoint", &self.is_waypoint())
            .field("refuel", &self.is_refuel())
            .finish()
    }
}

impl Serialize for SystemFlags {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("kind", &self.kind())?;
        map.serialize_entry("primary_kind", &self.primary_kind())?;
        map.serialize_entry("refuel", &self.is_refuel())?;
        map.serialize_entry("waypoint", &self.is_waypoint())?;
        map.end()
    }
}

impl SystemFlags {
    pub const fn new(primary_kind: StarKind, kind: StarKind) -> Self {
        Self(primary_kind.to_bits() | kind.to_bits() << StarKind::NUM_BITS)
    }

    pub const fn from_value(val: u8) -> Self {
        Self(val)
    }

    pub const fn value(self) -> u8 {
        self.0
    }

    pub fn set_waypoint(&mut self, val: bool) {
        const SHIFT: usize = 1 + (StarKind::NUM_BITS * 2);
        if val {
            self.0 |= u8::from(val) << SHIFT;
        } else {
            self.0 &= !(u8::from(val) << SHIFT);
        }
    }

    pub fn set_refuel(&mut self, val: bool) {
        const SHIFT: usize = StarKind::NUM_BITS * 2;
        if val {
            self.0 |= u8::from(val) << SHIFT;
        } else {
            self.0 &= !(u8::from(val) << SHIFT);
        }
    }

    pub const fn is_refuel(self) -> bool {
        const SHIFT: usize = StarKind::NUM_BITS * 2;
        self.0 & (1 << SHIFT) != 0
    }

    pub const fn is_waypoint(self) -> bool {
        const SHIFT: usize = 1 + (StarKind::NUM_BITS * 2);
        self.0 & (1 << SHIFT) != 0
    }

    pub fn has_scoopable(self) -> bool {
        self.primary_kind() == StarKind::Scoopable
            || self.kind() == StarKind::Scoopable
    }

    pub fn has_neutron(self) -> bool {
        self.primary_kind() == StarKind::Neutron
            || self.kind() == StarKind::Neutron
    }

    pub fn has_white_dwarf(self) -> bool {
        self.primary_kind() == StarKind::WhiteDwarf
            || self.kind() == StarKind::WhiteDwarf
    }

    pub const fn primary_kind(self) -> StarKind {
        let mask = (1 << (StarKind::NUM_BITS)) - 1;
        match self.0 & mask {
            0b00 => StarKind::Regular,
            0b01 => StarKind::Scoopable,
            0b10 => StarKind::WhiteDwarf,
            0b11 => StarKind::Neutron,
            _ => unreachable!(),
        }
    }

    pub const fn kind(self) -> StarKind {
        let mask = ((1 << (StarKind::NUM_BITS)) - 1) << StarKind::NUM_BITS;
        match (self.0 & mask) >> StarKind::NUM_BITS {
            0b00 => StarKind::Regular,
            0b01 => StarKind::Scoopable,
            0b10 => StarKind::WhiteDwarf,
            0b11 => StarKind::Neutron,
            _ => unreachable!(),
        }
    }

    pub const fn primary_mult(self) -> f32 {
        self.primary_kind().mult()
    }

    pub fn mult(self) -> f32 {
        self.primary_mult().max(self.kind().mult())
    }
}

mod flags {
    use serde::{Deserialize, Deserializer, Serializer};

    use super::SystemFlags;

    #[allow(clippy::trivially_copy_pass_by_ref)]
    pub fn serialize<S>(flags: &SystemFlags, ser: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        ser.serialize_u8(flags.value())
    }

    pub fn deserialize<'de, D>(deser: D) -> Result<SystemFlags, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(SystemFlags::from_value(<u8 as Deserialize>::deserialize(deser)?))
    }
}

// TODO: remove id, replace with router.id, benchmark
/// Node for [`rstar::RTree`]
#[repr(C, packed)]
#[derive(
    Debug,
    Clone,
    Copy,
    Serialize,
    Deserialize,
    Default,
    bytemuck::Pod,
    bytemuck::Zeroable,
)]
pub struct TreeNode {
    /// System flags
    #[serde(with = "flags")]
    pub flags: SystemFlags,
    /// Position in space
    pos: [f32; 3],
}


impl PartialEq for &TreeNode {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }
}

impl Eq for &TreeNode {}

impl PartialOrd for &TreeNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for &TreeNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.flags.cmp(&other.flags).then(
            self.pos()
                .iter()
                .zip(other.pos().iter())
                .map(|(a, b)| a.total_cmp(b))
                .reduce(|a, b| a.then(b))
                .unwrap_or_else(|| unreachable!()),
        )
    }
}

#[allow(clippy::inline_always)]
impl TreeNode {
    /// Retrieve matching [System] for this tree node
    pub fn get(&self, router: &Router) -> AstronavResult<System> {
        router.get(router.get_tree().id(self))
    }

    #[inline(always)]
    pub const fn pos(&self) -> &[f32; 3] {
        unsafe { &*(&raw const self.pos).cast::<[f32; 3]>() }
    }

    pub fn mult(&self) -> f32 {
        self.flags.mult()
    }

    pub const fn primary_mult(&self) -> f32 {
        self.flags.primary_mult()
    }

    pub fn range(&self, range2: f32) -> f32 {
        self.mult().powi(2) * range2
    }

    pub fn has_scoopable(&self) -> bool {
        self.flags.has_scoopable()
    }
}

impl From<[f32; 3]> for TreeNode {
    fn from(value: [f32; 3]) -> Self {
        Self { pos: value, flags: SystemFlags::default() }
    }
}

impl Hash for TreeNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.pos().iter().for_each(|v| v.to_ne_bytes().hash(state));
    }
}

impl sif_kdtree::Object for TreeNode {
    type Point = [f32; 3];

    fn position(&self) -> &Self::Point {
        self.pos()
    }
}

/// Star system info
#[derive(Debug, Clone)]
pub struct System {
    /// Unique System id
    pub id: u32,
    /// ID64 of star system
    pub id64: u64,
    /// Star system
    pub name: String,
    /// System flags
    pub flags: SystemFlags,
    /// Position
    pub pos: [f32; 3],
}

impl Serialize for System {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("id", &self.id)?;
        map.serialize_entry("id64", &self.id64)?;
        map.serialize_entry("name", &self.name)?;
        map.serialize_entry("name", &self.name)?;
        map.serialize_entry("kind", &self.flags.kind())?;
        map.serialize_entry("primary_kind", &self.flags.primary_kind())?;
        map.serialize_entry("refuel", &self.flags.is_refuel())?;
        map.serialize_entry("waypoint", &self.flags.is_waypoint())?;
        map.serialize_entry("pos", &self.pos)?;
        map.serialize_entry("tag", &self.tag())?;
        map.end()
    }
}

impl Display for System {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} [{}/#{}]", self.name, self.id, self.id64)
    }
}

impl System {
    pub fn edsm_link(&self) -> String {
        format!(
            "https://edsm.net/en/system?systemID64={id64}",
            id64 = self.id64
        )
    }
    pub(crate) fn to_node<'a>(&'a self, tree: &'a RouterTree) -> Result<&'a TreeNode> {
        tree.resolve(self)
    }

    pub fn dist2(&self, p: &[f32; 3]) -> f32 {
        dist2(&self.pos, p)
    }

    pub fn distp(&self, p: &Self) -> f32 {
        dist(&self.pos, &p.pos)
    }
    pub fn distp2(&self, p: &Self) -> f32 {
        self.dist2(&p.pos)
    }

    pub fn tag(&self) -> String {
        let primary_tag = match self.flags.primary_kind() {
            StarKind::Scoopable if self.flags.is_refuel() => 'f',
            other => other.tag(),
        }
        .to_uppercase();
        let tag = match self.flags.kind() {
            StarKind::Scoopable if self.flags.is_refuel() => 'f',
            other => other.tag(),
        };
        format!("{primary_tag}{tag}")
    }
}

impl Ord for System {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for System {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Serialize)]
pub struct RouteResult {
    pub hops: Vec<System>,
    pub time: f64,
}

#[derive(Debug, Serialize)]
pub struct ID64 {
    id64: u64,
    pub body_id: u16,
    pub sector: [u8; 3],
    pub boxel: [u8; 3],
    pub n2: u32,
    pub mass_code: u8,
    pub pos: [f32; 3],
}

#[allow(clippy::cast_possible_truncation)]
fn pos_from_sys_info(
    mass_code: u8,
    sector: [u8; 3],
    boxel: [u8; 3],
) -> [f32; 3] {
    const GALACTIC_ORIGIN: [f64; 3] = [-49985.0, -40985.0, -24105.0];
    const SECTOR_SIZE: f64 = 1280.0;
    let boxel_size: f64 = 10.0 * (2.0_f64).powi(mass_code.into());
    [
        (f64::from(boxel[0]).mul_add(
            boxel_size,
            f64::from(sector[0]).mul_add(SECTOR_SIZE, GALACTIC_ORIGIN[0]),
        ) + boxel_size / 2.0) as f32,
        (f64::from(boxel[1]).mul_add(
            boxel_size,
            f64::from(sector[1]).mul_add(SECTOR_SIZE, GALACTIC_ORIGIN[1]),
        ) + boxel_size / 2.0) as f32,
        (f64::from(boxel[2]).mul_add(
            boxel_size,
            f64::from(sector[2]).mul_add(SECTOR_SIZE, GALACTIC_ORIGIN[2]),
        ) + boxel_size / 2.0) as f32,
    ]
}

impl ID64 {
    pub const fn coords(&self) -> [f32; 3] {
        self.pos
    }
}

const fn take_bits(v: u64, bits: u8) -> (u64, u64) {
    let val = v & ((1 << bits) - 1);
    (val, v >> bits)
}

impl TryFrom<u64> for ID64 {
    type Error = eyre::Report;
    fn try_from(value: u64) -> Result<Self> {
        let original_id = value;
        let (mass_code, value) = take_bits(value, 3);
        let mass_code: u8 =
            mass_code.try_into().unwrap_or_else(|_| unreachable!());
        let box_coords_bits: u8 = 7 - mass_code;
        let n2_bits: u8 = 11 + mass_code * 3;
        let (z_box, value) = take_bits(value, box_coords_bits);
        let (z_sec, value) = take_bits(value, 7);
        let (y_box, value) = take_bits(value, box_coords_bits);
        let (y_sec, value) = take_bits(value, 6);
        let (x_box, value) = take_bits(value, box_coords_bits);
        let (x_sec, value) = take_bits(value, 7);
        let (n2, value) = take_bits(value, n2_bits);
        let (body_id, rest) = take_bits(value, 9);
        assert_eq!(rest, 0);
        let sector = [
            x_sec.try_into().unwrap_or_else(|_| unreachable!()),
            y_sec.try_into().unwrap_or_else(|_| unreachable!()),
            z_sec.try_into().unwrap_or_else(|_| unreachable!()),
        ];
        let boxel = [
            x_box.try_into().unwrap_or_else(|_| unreachable!()),
            y_box.try_into().unwrap_or_else(|_| unreachable!()),
            z_box.try_into().unwrap_or_else(|_| unreachable!()),
        ];
        Ok(Self {
            id64: original_id,
            body_id: body_id.try_into()?,
            sector,
            boxel,
            n2: n2.try_into()?,
            mass_code,
            pos: pos_from_sys_info(mass_code, sector, boxel),
        })
    }
}

#[derive(Debug, Serialize)]
pub struct VisitedSystem {
    id: ID64,
    visit_count: Option<u32>,
    last_visit: Option<u32>,
}

#[derive(Debug, Serialize)]
pub struct VisitedStarsCache {
    recent: bool,
    num_entries: u32,
    account_id: u32,
    cmdr_id: u64,
    entries: Vec<VisitedSystem>,
}

pub fn load_visited(path: &PathBuf) -> AstronavResult<VisitedStarsCache> {
    use eyre::eyre;
    const END_MARKER: u64 = 0x5AFE_C0DE_5AFE_C0DE;
    const MAGIC: &[u8; 12] = b"VisitedStars";
    let mut read_buf = vec![0; MAGIC.len()];
    let mut fh = BufReader::new(File::open(path)?);
    fh.read_exact(&mut read_buf)?;
    if read_buf != MAGIC {
        return Err(AstronavError::Other(eyre!(
            "Invalid magic found in VisitedStarsCache"
        )));
    }
    let recent = fh.read_u32::<LittleEndian>()?;
    let version = fh.read_u32::<LittleEndian>()?;
    let start = fh.read_u32::<LittleEndian>()?;
    let num_entries = fh.read_u32::<LittleEndian>()?;
    let entry_len = fh.read_u32::<LittleEndian>()?;
    let account_id = fh.read_u32::<LittleEndian>()?;
    let _padding = fh.read_u32::<LittleEndian>()?;
    let cmdr_id = fh.read_u64::<LittleEndian>()?;
    if (MAGIC.len() + 4 * 7 + 8) != (start as usize) {
        return Err(AstronavError::Other(eyre!("Invalid start offset")));
    }
    let mut entries = vec![];
    let recent = match version {
        100 => {
            if entry_len != 8 {
                return Err(AstronavError::Other(eyre!(
                    "Invalid entry_length"
                )));
            }
            loop {
                let id = fh.read_u64::<LittleEndian>()?;
                if id == END_MARKER {
                    break;
                }
                entries.push(VisitedSystem {
                    id: ID64::try_from(id)?,
                    visit_count: None,
                    last_visit: None,
                });
            }
            recent == 0x7f00
        }
        200 => {
            if entry_len != 16 {
                return Err(AstronavError::Other(eyre!(
                    "Invalid entry_length"
                )));
            }
            loop {
                let id = fh.read_u64::<LittleEndian>()?;
                if id == END_MARKER {
                    break;
                }
                let visit_count = Some(fh.read_u32::<LittleEndian>()?);
                let last_visit = Some(fh.read_u32::<LittleEndian>()?);
                entries.push(VisitedSystem {
                    id: ID64::try_from(id)?,
                    visit_count,
                    last_visit,
                });
            }
            recent == 0x200
        }
        version => {
            return Err(AstronavError::Other(eyre!(
                "Don't know how to parse VisitedStarsCache version={version}"
            )));
        }
    };
    Ok(VisitedStarsCache { recent, num_entries, account_id, cmdr_id, entries })
}

#[derive(Clone)]
pub struct ResolvedSystem(pub SysEntry, pub System);
impl PartialEq for ResolvedSystem {
    fn eq(&self, other: &Self) -> bool {
        self.1.id == other.1.id
    }
}

impl PartialOrd for ResolvedSystem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for ResolvedSystem {}

impl Ord for ResolvedSystem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.1.id.cmp(&other.1.id)
    }
}
