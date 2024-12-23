#![cfg(feature = "pyo3")]
use std::{
    io::BufWriter,
    ops::ControlFlow,
    path::PathBuf,
    sync::{atomic::AtomicBool, Arc},
    thread::scope,
    time::{Duration, Instant},
};

use byteorder::{LittleEndian, WriteBytesExt};
use color_eyre::eyre::Result;
use common::RouteResult;
use crossbeam_channel::{bounded, RecvTimeoutError};
use human_repr::{HumanCount, HumanDuration, HumanThroughput};
use itertools::Itertools;
use pyo3::{exceptions::PyKeyboardInterrupt, prelude::*, types::PyDict};
use pythonize::{depythonize, pythonize};
use rustc_hash::FxHashMap;
use tracing::*;

use crate::{
    common::{dist, AstronavError, SysEntry, System},
    event::Event,
    route::Router,
    ship::{NamedShip, Ship},
    *,
};

#[pyclass(dict, name = "Router")]
#[derive(Debug)]
struct PyRouter(Router);

#[pymethods]
impl PyRouter {
    #[new]
    #[pyo3(signature = (callback = None, status_interval = 0.0))]
    fn new(callback: Option<PyObject>, status_interval: f64) -> Self {
        let callback = Box::new(move |router: &Router, event: &Event| {
            Python::with_gil(|py| {
                if let Err(error) = py.check_signals() {
                    if error.is_instance_of::<PyKeyboardInterrupt>(py) {
                        router.interrupt();
                    }
                };
                match callback.as_ref() {
                    Some(cb) => {
                        let res = cb
                            .call(py, (pythonize(py, event)?,), None)?
                            .extract::<Option<bool>>(py)?
                            .unwrap_or_default();
                        if res {
                            router.interrupt();
                        }
                        Ok(())
                    }
                    None => {
                        info!("{event}");
                        Ok(())
                    }
                }
            })
        });
        let mut router = Router::new(Duration::from_secs_f64(status_interval));
        router.set_callback(callback);
        Self(router)
    }

    #[pyo3(signature = (flag))]
    fn set_deterministic(&mut self, flag: bool) -> PyResult<()> {
        self.0.set_deterministic(flag);
        Ok(())
    }

    #[pyo3(signature = (callback))]
    fn set_callback(&mut self, callback: PyObject) -> PyResult<()> {
        let callback = Box::new(move |router: &Router, event: &Event| {
            Python::with_gil(|py| {
                py.check_signals()?;
                if let Err(error) = py.check_signals() {
                    if error.is_instance_of::<PyKeyboardInterrupt>(py) {
                        router.interrupt();
                    }
                };
                let res = callback
                    .call(py, (pythonize(py, event)?,), None)?
                    .extract::<Option<bool>>(py)?
                    .unwrap_or_default();
                if res {
                    router.interrupt();
                }
                Ok(())
            })
        });
        self.0.set_callback(callback);
        Ok(())
    }

    fn preprocess_edsm(
        &self,
        _bodies_path: String,
        _systems_path: String,
        _out_path: String,
    ) -> PyResult<()> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Please use Spansh's Galaxy dump and preprocess_galaxy()",
        ))
    }

    #[pyo3(signature=(path, out_path, load_immediately = false))]
    fn preprocess_galaxy(
        &mut self,
        path: Option<PathBuf>,
        out_path: PathBuf,
        load_immediately: bool,
    ) -> Result<()> {
        galaxy::process_galaxy_dump(path.as_ref(), &out_path, &self.0)?;
        Ok(())
    }

    #[pyo3(signature = (path, immediate=false, mmap_tree=true))]
    fn load(&mut self, path: PathBuf, immediate: bool, mmap_tree: bool) -> Result<()> {
        self.0.set_path(&path)?;
        if immediate {
            self.0.load(map_tree, &[], 0.0)?;
        }
        Ok(())
    }

    fn unload(&mut self, py: Python<'_>) -> PyObject {
        self.0.unload();
        py.None()
    }

    #[pyo3(signature = (ship))]
    fn set_ship(&mut self, py: Python<'_>, ship: PyShip) -> PyObject {
        self.0.set_ship(ship.get_ship().into());
        py.None()
    }

    fn plot(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let mut max_v = [0f32, 0f32, 0f32];
        let mut min_v = [0f32, 0f32, 0f32];
        for node in self.0.get_tree().iter() {
            let p = node.pos();
            for i in 0..3 {
                if p[i] > max_v[i] {
                    max_v[i] = p[i];
                }
                if p[i] < min_v[i] {
                    min_v[i] = p[i];
                }
            }
        }
        let plot_bbox: ((f32, f32), (f32, f32)) =
            ((min_v[0], max_v[0]), (min_v[2], max_v[2]));
        Ok(plot_bbox.to_object(py))
    }

    fn gz_test(&self, path: PathBuf, py: Python<'_>) -> PyResult<PyObject> {
        use std::{
            fs::File,
            io::{BufRead, BufReader},
        };

        info!("RUN!");
        use flate2::read::GzDecoder;
        let mut buf = String::new();
        let mut reader = BufReader::new(GzDecoder::new(File::open(path)?));
        let total_len = reader.get_ref().get_ref().metadata()?.len() as f64;
        let t_start = Instant::now();
        while reader.read_line(&mut buf)? != 0 {
            buf.clear();
        }
        let dt = t_start.elapsed();
        let rate = (total_len / dt.as_secs_f64()).human_throughput_bytes();
        println!(
            "{} in {}: {}",
            total_len.human_count_bytes(),
            dt.human_duration(),
            rate
        );
        Ok(py.None())
    }

    fn test(&mut self, range: f32, py: Python<'_>) -> PyResult<PyObject> {
        use std::fs::File;
        let r2 = range.powi(2);
        let tree = self.0.get_tree();
        let mut edges: usize = 0;
        let mut nodes: usize = 0;
        let mut of = BufWriter::new(File::create("nbs.bin")?);
        let mut max_diff = 0;
        for node in tree.iter() {
            nodes += 1;
            let mut nb_ids = tree
                .locate_within_distance(*node.pos(), r2)
                .map(|nb| nb.id)
                .collect_vec();
            nb_ids.sort();
            max_diff = max_diff.max(
                nb_ids
                    .windows(2)
                    .map(|w| match w {
                        [a, b] => b - a,
                        _ => 0,
                    })
                    .max()
                    .unwrap_or_default(),
            );
            edges += nb_ids.len();
            of.write_u32::<LittleEndian>(node.id)?;
            of.write_u64::<LittleEndian>(nb_ids.len() as u64)?;
            nb_ids
                .into_iter()
                .map(|id| of.write_u32::<LittleEndian>(id))
                .collect::<Result<Vec<_>, _>>()?;
            if nodes % 100_000 == 0 {
                let deg = (edges as f64) / (nodes as f64);
                info!(
                    "Nodes: {nodes}, Edges: {edges}, Average dregree: {deg}, Max diff: {max_diff}"
                )
            }
        }
        let deg = (edges as f64) / (nodes as f64);
        info!("Done! Nodes: {nodes}, Edges: {edges}, Average dregree: {deg}");
        Ok(py.None())
    }

    fn nb_perf_test(
        &mut self,
        range: f32,
        py: Python<'_>,
    ) -> PyResult<PyObject> {
        use rand::{Rng, SeedableRng};
        let tree = self.0.get_tree();
        let total_nodes = tree.len();
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let mut num_loops: usize = 0;
        let num_loops_inner: usize = 1_000;
        let mut d_total = 0.0;
        for _ in 0..100 {
            let node = *self
                .0
                .get_tree()
                .iter()
                .nth(rng.gen_range(0..total_nodes))
                .unwrap_or_else(|| unreachable!());
            let t_start = Instant::now();
            let mut cnt = 0;
            for _ in 0..num_loops_inner {
                cnt += self
                    .0
                    .neighbours(&node, range, |_| ControlFlow::Continue(()));
            }
            let dt = t_start.elapsed().as_secs_f64() / num_loops_inner as f64;
            num_loops += 1;
            d_total += dt;
            info!(
                "Took: {}, Avg: {}",
                dt.human_duration(),
                (d_total / num_loops as f64).human_duration()
            );
            if num_loops == (dt as usize) {
                return Ok(cnt.into_py(py));
            }
        }
        Ok(py.None())
    }

    fn get(&mut self, id: u32, py: Python<'_>) -> PyResult<PyObject> {
        Ok(pythonize(py, &self.0.get(id)?)?)
    }

    fn galaxy_grid_test(&self) -> PyResult<()> {
        use common::TreeNode;
        let t_load = Instant::now();
        info!("Loading [{}]", self.0.path.display());
        let (systems, total_nodes) = data_loader::load(&self.0.path, &[], 0.0)?;
        let rate = (total_nodes as f64) / t_load.elapsed().as_secs_f64();
        info!(
            "{} Systems loaded in {:.2}, {:.2}",
            systems.len(),
            t_load.elapsed().human_duration(),
            rate.human_throughput(" systems")
        );

        let mut grid: FxHashMap<(i32, i32, i32), Vec<TreeNode>> =
            FxHashMap::default();
        for node in systems.into_iter() {
            let p = node.pos();
            let id = (
                (p[0] / 1280.0).round() as i32,
                (p[1] / 1280.0).round() as i32,
                (p[2] / 1280.0).round() as i32,
            );
            grid.entry(id).or_default().push(node);
        }
        let mut stats =
            grid.into_iter().map(|(k, v)| (k, v.len())).collect_vec();
        stats.sort_by_key(|(_, v)| *v);
        for (k, v) in &stats {
            println!("{k:?} {v}")
        }
        println!("Total cells: {}", stats.len());
        Ok(())
    }

    fn best_permutation(
        &self,
        hops: PyObject,
        py: Python<'_>,
    ) -> PyResult<Vec<PyObject>> {
        use std::sync::atomic::Ordering;

        use common::ResolvedSystem;
        fn score(nodes: &[ResolvedSystem]) -> f32 {
            if nodes.len() < 2 {
                return f32::INFINITY;
            }
            let mut ret = 0.0;
            for win in nodes.windows(2) {
                if let [a, b] = win {
                    ret += common::dist2(&a.1.pos, &b.1.pos);
                }
            }
            ret
        }
        let hops: Vec<SysEntry> = depythonize(hops.as_ref(py))?;
        info!("Finding shortest permutation, this may take a bit...");
        let systems = self.0.resolve(&hops)?;
        let mut route: Vec<ResolvedSystem> = Vec::with_capacity(hops.len());
        for (hop, sys) in hops.iter().cloned().zip(systems.into_iter()) {
            let hop = hop.clone();
            let sys = sys.ok_or_else(|| {
                AstronavError::SystemNotFoundError(hop.clone())
            })?;
            route.push(ResolvedSystem(hop, sys));
        }
        info!("Starting simulated annealing...");

        let best = scope(|s| {
            let (tx, rx) = bounded(1024 * 8);
            let stop = Arc::new(AtomicBool::new(false));
            let handles = (0..num_cpus::get())
                .map(|_| {
                    let stop = stop.clone();
                    let tx = tx.clone();
                    let mut route = route.clone();
                    let mut best = (f32::INFINITY, vec![]);
                    s.spawn(move || {
                        use rand::{
                            distributions::Distribution, rngs::SmallRng, Rng,
                            SeedableRng,
                        };
                        let mut t_send = Instant::now();
                        let mut rng = SmallRng::from_entropy();
                        let rand_dist =
                            rand::distributions::Uniform::new(0, route.len());
                        let mut n_checked: usize = 0;
                        let k_max_n: usize = 10_000_000;
                        let k_max = k_max_n as f32;
                        loop {
                            for k in 0..k_max_n {
                                let temperature = 1.0 - ((k + 1) as f32) / k_max;
                                if stop.load(Ordering::SeqCst) {
                                    if let Err(e) = tx.send((best.clone(), n_checked, temperature)) {
                                        panic!("Failed to send best value through channel: {e}");
                                    };
                                    return;
                                }
                                let idx_1 = rand_dist.sample(&mut rng);
                                let idx_2 = rand_dist.sample(&mut rng);
                                let prev_score = score(&route);
                                route.swap(idx_1, idx_2);
                                let new_score = score(&route);
                                n_checked += 1;
                                let p = (-(new_score - prev_score) / temperature)
                                    .exp().max(0.0)
                                    as f64;
                                if (new_score < prev_score) || rng.gen_bool(p) {
                                    if new_score < best.0 {
                                        best.0 = new_score;
                                        best.1 = route.to_vec();
                                        if t_send.elapsed().as_secs_f64() > 1.0 {
                                            if let Err(e) = tx.send((best.clone(), n_checked, temperature)) {
                                                panic!("Failed to send best value through channel: {e}");
                                            }
                                            t_send = Instant::now();
                                            n_checked = 0;
                                        }
                                    }
                                } else {
                                    route.swap(idx_1, idx_2);
                                }
                            }
                        }
                    })
                })
                .collect_vec();
            drop(tx);
            let mut best = (f32::INFINITY, vec![]);
            let t_start = Instant::now();
            let mut n_checks = 0;
            let dt = Duration::from_secs_f64(60.0);
            loop {
                match rx.recv_timeout(dt) {
                    Ok(((score, route), n, temp)) => {
                        n_checks += n;
                        if score < best.0 {
                            best.0 = score;
                            best.1 = route;
                            let rate = (n_checks as f64)
                                / t_start.elapsed().as_secs_f64();
                            info!(
                                "new best: {score}, {rate}, Temperature: {temp}",
                                rate = rate.human_throughput(" checks")
                            );
                        }
                    }
                    Err(RecvTimeoutError::Timeout) => {
                        stop.store(true, Ordering::SeqCst);
                        warn!(
                            "No improvement in the last {}",
                            dt.human_duration()
                        );
                        break;
                    }
                    Err(RecvTimeoutError::Disconnected) => {
                        break;
                    }
                }
            }
            if let Err(e) = handles
                .into_iter()
                .map(|h| h.join())
                .collect::<Result<Vec<_>, _>>()
            {
                panic!("Failed to join threads: {e:?}")
            };
            best
        });
        Ok(best
            .1
            .into_iter()
            .map(|ResolvedSystem(h, sys)| pythonize(py, &(h, sys.id)))
            .collect::<Result<Vec<_>, _>>()?)
    }

    #[pyo3(
        signature = (hops, range, mode=None, radius=0.0, detailed=false)
    )]
    fn route(
        &mut self,
        hops: Vec<PyObject>,
        range: Option<f32>,
        mode: Option<PyObject>,
        radius: f32,
        detailed: bool,
        mmap_tree: bool,
        py: Python<'_>,
    ) -> PyResult<PyObject> {
        let hops: Vec<SysEntry> = hops
            .into_iter()
            .map(|h| depythonize(h.as_ref(py)))
            .collect::<Result<Vec<SysEntry>, _>>()?;
        let (is_default_mode, mode) = (mode
            .map(|m| depythonize(m.as_ref(py)))
            .transpose()?)
        .map_or_else(|| (true, Default::default()), |mode| (false, mode));
        let mut ids: Vec<u32> = vec![];
        for (res, hop) in self.0.resolve(&hops)?.into_iter().zip(hops) {
            match res {
                Some(sys) => {
                    if !matches!(hop, SysEntry::ID(_)) {
                        info!("{hop} => {sys}");
                    }
                    ids.push(sys.id);
                }
                None => Err(AstronavError::SystemNotFoundError(hop))?,
            }
        }
        if is_default_mode {
            warn!("No mode specified, defaulting to {}", mode);
        }
        match self.0.compute_route(&ids, range, radius, mode, mmap_tree) {
            Ok((dt, nodes)) => {
                info!("Route computed in {}", dt.human_duration());
                if !nodes.is_empty() {
                    let route_distance: f32 = nodes
                        .windows(2)
                        .map(|w| dist(&w[0].pos, &w[1].pos))
                        .sum();
                    info!(
                        "Destination reached in {} jumps, {} Ly",
                        nodes.len(),
                        route_distance
                    );
                }
                if detailed {
                    Ok(Python::with_gil(|py| {
                        pythonize(
                            py,
                            &RouteResult {
                                hops: nodes,
                                time: dt.as_secs_f64(),
                            },
                        )
                    })?)
                } else {
                    Ok(Python::with_gil(|py| pythonize(py, &nodes))?)
                }
            }
            Err(AstronavError::PyError(e)) => Err(e),
            Err(err) => Err(err.into()),
        }
    }

    #[pyo3(
        signature = (hops)
    )]
    fn resolve(
        &self,
        hops: Vec<PyObject>,
        py: Python<'_>,
    ) -> PyResult<PyObject> {
        let hops: Vec<SysEntry> = hops
            .into_iter()
            .map(|h| depythonize(h.as_ref(py)))
            .collect::<Result<Vec<SysEntry>, _>>()?;
        let systems = self.0.resolve(&hops)?;
        let ret: Vec<(SysEntry, Option<System>)> = hops
            .into_iter()
            .zip(systems.iter())
            .map(|(id, sys)| (id, sys.clone()))
            .collect();
        for (k, v) in &ret {
            v.as_ref().map_or_else(
                || info!("{k} => <NOT_FOUND>"),
                |v| info!("{k} => {v}"),
            );
        }
        Ok(PyDict::from_sequence(py, pythonize(py, &ret)?)?.to_object(py))
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{}", &self.0))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", &self))
    }
}

#[pyclass(dict, name = "Ship")]
#[derive(Debug, Clone)]
struct PyShip {
    ship: NamedShip,
}

#[pymethods]
impl PyShip {
    #[new]
    fn new(
        base_mass: f32,
        fuel_mass: f32,
        fuel_capacity: f32,
        fsd_type: (char, u8),
        max_fuel: f32,
        opt_mass: f32,
        guardian_booster: usize,
    ) -> PyResult<Self> {
        let ship = NamedShip {
            ship: Ship::new(
                base_mass,
                fuel_mass,
                fuel_capacity,
                fsd_type,
                max_fuel,
                opt_mass,
                guardian_booster,
            )?,
            ship_type: "<None>".to_string(),
            ident: "<None>".to_string(),
            ship_name: None,
        };
        Ok(Self { ship })
    }

    #[staticmethod]
    fn from_loadout(py: Python<'_>, loadout: String) -> PyResult<Self> {
        Ok(Self { ship: Ship::new_from_json(&loadout)? })
    }

    #[staticmethod]
    fn from_journal(py: Python<'_>) -> PyResult<PyObject> {
        let ships: Vec<(PyObject, PyObject)> = Ship::new_from_journal()?
            .into_iter()
            .map(|ship| {
                let ship_name_py = format!("{ship}").to_object(py);
                let ship_py = (Self { ship }).into_py(py);
                (ship_name_py, ship_py)
            })
            .collect();
        Ok(PyDict::from_sequence(py, ships.to_object(py))?.to_object(py))
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(pythonize(py, &self.ship)?)
    }

    fn fuel_cost(&self, dist: f32) -> f32 {
        self.ship.fuel_cost(dist)
    }

    #[getter]
    fn range(&self) -> f32 {
        self.ship.range()
    }

    #[getter]
    fn max_range(&self) -> f32 {
        self.ship.max_range()
    }

    fn make_jump(&mut self, dist: f32) -> Option<f32> {
        self.ship.make_jump(dist)
    }

    fn can_jump(&self, dist: f32) -> bool {
        self.ship.can_jump(dist)
    }

    #[pyo3(signature=(fuel_amount = None))]
    fn refuel(&mut self, fuel_amount: Option<f32>) {
        if let Some(fuel) = fuel_amount {
            self.ship.fuel_mass =
                (self.ship.fuel_mass + fuel).min(self.ship.fuel_capacity)
        } else {
            self.ship.fuel_mass = self.ship.fuel_capacity;
        }
    }

    #[pyo3(signature=(factor))]
    fn boost(&mut self, factor: f32) {
        self.ship.boost(factor);
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{}", &self.ship))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", &self.ship))
    }
}

impl PyShip {
    fn get_ship(&self) -> NamedShip {
        self.ship.clone()
    }
}

#[pyfunction(signature=(path=None))]
pub fn load_visited(
    path: Option<PathBuf>,
    py: Python<'_>,
) -> PyResult<PyObject> {
    let path = path.unwrap_or_else(|| todo!());
    Ok(pythonize(py, &common::load_visited(&path)?)?)
}

#[pymodule]
pub fn astronav(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    color_eyre::install()?;
    pyo3_log::init();

    m.add_class::<PyRouter>()?;
    m.add_class::<PyShip>()?;
    m.add_wrapped(pyo3::wrap_pyfunction!(load_visited))?;
    #[cfg(feature = "edmc_plugin")]
    edmc_plugin::init(py, m)?;
    Ok(())
}
