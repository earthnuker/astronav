use std::{
    num::NonZeroUsize, path::PathBuf, thread::JoinHandle, time::Duration,
};

use color_eyre::eyre::{anyhow, Report, Result};
use crossbeam_channel::{unbounded, Receiver, Sender};
use eframe::{
    egui::{self, TextureHandle, Ui, Visuals},
    NativeOptions,
};
use egui_plot::{Line, Plot, PlotPoints};
use egui_tracing::tracing::collector::EventCollector;
use indoc::{formatdoc, indoc};
use strum::IntoEnumIterator;
use tracing::*;
use tracing_subscriber::{
    fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Layer,
};

use crate::{
    common::{AstronavResult, BeamWidth, RelativeTime, System, F32},
    event::Event,
    route::{ModeConfig, RefuelMode, RouteMode, Router, ShipMode},
    ship::{NamedShip, Ship},
};

mod galaxy_map;
struct UiState {
    ships: Option<Vec<NamedShip>>,
    selected_ship: Option<NamedShip>,
    show_logs: bool,
    mode_name: Option<RouteMode>,
    mode_config: Option<ModeConfig>,
    last_error: Option<Report>,
    path: Option<PathBuf>,
    running: bool,
    galaxy_map: Option<TextureHandle>,
}
impl UiState {
    const fn new() -> Self {
        Self {
            show_logs: false,
            mode_name: None,
            mode_config: None,
            ships: None,
            selected_ship: None,
            last_error: None,
            running: false,
            path: None,
            galaxy_map: None,
        }
    }
}

#[derive(Debug)]
enum RouterReturn {
    None,
    Route(Vec<System>),
    Event(Event),
}

type Command = Box<dyn FnOnce(&mut Router) -> Result<RouterReturn> + Send>;

struct AstronavGUI {
    router_handle: JoinHandle<()>,
    rx: Receiver<Result<RouterReturn>>,
    tx: Sender<Command>,
    collector: EventCollector,
    ui_state: UiState,
}

impl AstronavGUI {
    fn new(
        cc: &eframe::CreationContext<'_>,
        router: Router,
        collector: EventCollector,
    ) -> Self {
        cc.egui_ctx
            .set_visuals(Visuals { button_frame: true, ..Visuals::dark() });
        let (result_tx, result_rx) = unbounded::<Result<RouterReturn>>();
        let (command_tx, command_rx) = unbounded::<Command>();
        let router_handle = std::thread::spawn(move || {
            let mut router = router;
            router.status_interval = Duration::from_secs_f32(0.1);
            let callback_tx = result_tx.clone();
            router.set_callback(Box::new(move |router, event| {
                callback_tx.send(Ok(RouterReturn::Event(event.clone())))?;
                info!("{event}");
                Ok(())
            }));

            for cmd in command_rx {
                if let Err(e) = result_tx.send(cmd(&mut router)) {
                    warn!("Failed to send result: {e}");
                };
            }
        });
        Self {
            router_handle,
            tx: command_tx,
            rx: result_rx,
            collector,
            ui_state: UiState::new(),
        }
    }

    fn exec<F>(&self, cmd: F)
    where
        F: FnOnce(&mut Router) -> Result<RouterReturn> + Send + 'static,
    {
        if let Err(e) = self.tx.send(Box::new(cmd)) {
            error!("Failed to send command to router: {e}");
        }
    }

    fn default_mode(&self) -> Option<ModeConfig> {
        self.ui_state.mode_name.as_ref().map(|mode| match mode {
            RouteMode::BeamSearch => ModeConfig::BeamSearch {
                beam_width: BeamWidth::default(),
                refuel_mode: Some(RefuelMode::WhenEmpty),
                refuel_primary: true,
                boost_primary: true,
                range_limit: f32::INFINITY,
            },
            RouteMode::DepthFirst => ModeConfig::DepthFirst,
            RouteMode::IncrementalBroadening => {
                ModeConfig::IncrementalBroadening
            }
            RouteMode::BeamStack => ModeConfig::BeamStack,
            RouteMode::AStar => ModeConfig::AStar { weight: F32(1.0) },
            RouteMode::Dijkstra => ModeConfig::Dijkstra,
            RouteMode::IncrementalBeamSearch => {
                ModeConfig::IncrementalBeamSearch { beam_width: 1024 }
            }
            RouteMode::Ship => ModeConfig::Ship { ship_mode: ShipMode::Jumps },
        })
    }

    fn get_mode(&self) -> Option<RouteMode> {
        self.ui_state.mode_config.as_ref().map(|mode| match mode {
            ModeConfig::BeamSearch { .. } => RouteMode::BeamSearch,
            ModeConfig::DepthFirst => RouteMode::DepthFirst,
            ModeConfig::IncrementalBroadening => {
                RouteMode::IncrementalBroadening
            }
            ModeConfig::BeamStack => RouteMode::BeamStack,
            ModeConfig::AStar { .. } => RouteMode::AStar,
            ModeConfig::Dijkstra => RouteMode::Dijkstra,
            ModeConfig::IncrementalBeamSearch { .. } => {
                RouteMode::IncrementalBeamSearch
            }
            ModeConfig::Ship { .. } => RouteMode::Ship,
        })
    }

    fn set_default_mode(&mut self) {
        if self.get_mode().as_ref() == self.ui_state.mode_name.as_ref() {
            return;
        }
        self.ui_state.mode_config = self.default_mode();
    }

    fn options_refuel_mode(ui: &mut Ui, refuel_mode: &mut Option<RefuelMode>) {
        egui::ComboBox::from_label("Refuel Mode")
            .width(200.0)
            .selected_text(match refuel_mode {
                Some(RefuelMode::WhenPossible) => "When possible",
                Some(RefuelMode::WhenEmpty) => "When empty",
                Some(RefuelMode::LeastJumps) => "Least amount of jumps",
                None => "<None>",
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(refuel_mode, None, "<None>");
                ui.selectable_value(
                    refuel_mode,
                    Some(RefuelMode::WhenPossible),
                    "When possible",
                );
                ui.selectable_value(
                    refuel_mode,
                    Some(RefuelMode::WhenEmpty),
                    "When empty",
                );
                ui.selectable_value(
                    refuel_mode,
                    Some(RefuelMode::LeastJumps),
                    "Least amount of jumps",
                );
            });
    }

    fn options_beam_width(ui: &mut Ui, beam_width: &mut BeamWidth) {
        #[derive(Debug, PartialEq, Eq)]
        enum BeamType {
            Infinite,
            Fraction,
            Absolute,
        }
        let mut beam_type = match beam_width {
            BeamWidth::Absolute(_) => BeamType::Absolute,
            BeamWidth::Fraction(_, _) => BeamType::Fraction,
            BeamWidth::Infinite => BeamType::Infinite,
        };
        egui::ComboBox::from_label("Beam Mode")
            .width(200.0)
            .selected_text(match beam_type {
                BeamType::Infinite => "Infinite",
                BeamType::Fraction => "Fraction",
                BeamType::Absolute => "Aboslute",
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(
                    &mut beam_type,
                    BeamType::Infinite,
                    "Infinite",
                ).on_hover_text("Infinite beam width, guaranteed to find the shortest path if one exists, very slow");
                ui.selectable_value(
                    &mut beam_type,
                    BeamType::Absolute,
                    "Absolute",
                ).on_hover_text("Fixed beam width, tradeoff between speed and route efficientcy, 1000*(number of cpu cores) is a good starting value");
                ui.selectable_value(
                    &mut beam_type,
                    BeamType::Fraction,
                    "Fraction",
                ).on_hover_text("Fractional beam width depending on number of neighbors, might be good in some cases i don't know");
            });
        *beam_width = match beam_type {
            BeamType::Infinite => BeamWidth::Infinite,
            BeamType::Fraction => {
                if !matches!(beam_width, BeamWidth::Fraction(_, _)) {
                    *beam_width = BeamWidth::Fraction(
                        1,
                        NonZeroUsize::try_from(1)
                            .unwrap_or_else(|_| unreachable!()),
                    );
                }
                let BeamWidth::Fraction(numerator, denominator) = beam_width
                else {
                    return;
                };
                let mut numerator = format!("{numerator}");
                let mut denominator = format!("{denominator}");
                ui.text_edit_singleline(&mut numerator);
                ui.text_edit_singleline(&mut denominator);
                let (Ok(mut numerator), Ok(denominator)) = (
                    numerator.parse::<usize>(),
                    denominator.parse::<NonZeroUsize>(),
                ) else {
                    return;
                };
                numerator = numerator.min(denominator.get());
                BeamWidth::Fraction(numerator, denominator)
            }
            BeamType::Absolute => {
                if !matches!(beam_width, BeamWidth::Absolute(_)) {
                    *beam_width = BeamWidth::Absolute(1000 * num_cpus::get());
                }
                let BeamWidth::Absolute(width) = beam_width else {
                    unreachable!();
                };
                let mut width = width.next_power_of_two().trailing_zeros();
                ui.add(
                    egui::Slider::new(&mut width, 0..=(usize::BITS - 1))
                        .logarithmic(true)
                        .text("Beam Width")
                        .custom_parser(|n| {
                            n.parse()
                                .map(|n: usize| {
                                    (usize::BITS - n.leading_zeros())
                                        .saturating_sub(1)
                                        as f64
                                })
                                .ok()
                        })
                        .custom_formatter(|n: f64, _| {
                            format!("{}", 1usize << (n as u32))
                        }),
                );
                BeamWidth::Absolute(1 << width)
            }
        }
    }

    fn options_beam(&mut self, ui: &mut Ui) {
        let Some(ModeConfig::BeamSearch {
            refuel_primary,
            boost_primary,
            refuel_mode,
            beam_width,
            range_limit,
        }) = self.ui_state.mode_config.as_mut()
        else {
            unreachable!();
        };
        Self::options_beam_width(ui, beam_width);
        Self::options_refuel_mode(ui, refuel_mode);
        ui.checkbox(refuel_primary, "Refuel only from primary stars");
        ui.checkbox(boost_primary, "Boost only from primary stars");
    }

    fn options_astar(&mut self, ui: &mut Ui) {
        let Some(ModeConfig::AStar { weight }) =
            self.ui_state.mode_config.as_mut()
        else {
            unreachable!();
        };
        egui::Grid::new("AStar").show(ui, |ui| {
            ui.label("Weight");
            ui.add(
                egui::DragValue::new(&mut weight.0)
                    .speed(0.1)
                    .clamp_range(0.0..=f32::INFINITY),
            )
            .on_hover_text(indoc! {"
                A*-Search weight
                - 0.0 is equivalent to Dijkstra
                - 1.0 is default A*
                - Infinite means greedy search
            "});
        });
    }

    fn options_ship(&mut self, ui: &mut Ui) {
        const SAMPLES: usize = 10000;
        let ships = self.ui_state.ships.get_or_insert_with(|| {
            match Ship::new_from_journal() {
                Ok(res) => res,
                Err(e) => {
                    error!("Failed to load ships: {e}");
                    vec![]
                }
            }
        });
        let Some(ModeConfig::Ship { ship_mode }) =
            self.ui_state.mode_config.as_mut()
        else {
            unreachable!()
        };
        let selected = &mut self.ui_state.selected_ship;
        egui::ComboBox::from_label("Ship")
            .width(200.0)
            .selected_text(
                selected.as_ref().map_or_else(
                    || "<None>".to_string(),
                    |ship| format!("{ship}"),
                ),
            )
            .show_ui(ui, |ui| {
                for ship in ships {
                    ui.selectable_value(
                        selected,
                        Some(ship.clone()),
                        format!("{ship}"),
                    );
                }
            });
        let Some(ship) = selected else {
            return;
        };

        egui::ComboBox::from_label("Ship Mode")
            .width(200.0)
            .selected_text(match ship_mode {
                ShipMode::Fuel => "Fuel",
                ShipMode::Jumps => "Jumps",
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(ship_mode, ShipMode::Fuel, "Fuel")
                    .on_hover_text("Minimize amount of fuel used");
                ui.selectable_value(ship_mode, ShipMode::Jumps, "Jumps")
                    .on_hover_text("Minimize number of jumps");
            });
        ui.collapsing("FSD Profile", |ui| {
            let points: PlotPoints = (0..=SAMPLES)
                .map(|n| {
                    let n = ((n as f64) / (SAMPLES as f64)) as f32;
                    [
                        (n * ship.fuel_capacity) as f64,
                        ship.jump_range(n * ship.fuel_capacity, true) as f64,
                    ]
                })
                .collect();
            let line = Line::new(points);
            Plot::new("FSD Profile")
                .set_margin_fraction([0.1, 0.1].into())
                .width(600.0)
                .height(300.0)
                .label_formatter(|_, val| {
                    format!("{:.2} Ly @ {:.2} t", val.y, val.x)
                })
                .show(ui, |plot_ui| plot_ui.line(line));
        });
        self.ui_state.selected_ship = selected.take();
        self.ui_state.mode_config =
            Some(ModeConfig::Ship { ship_mode: *ship_mode });
    }

    const fn has_options(&self, mode: &RouteMode) -> bool {
        matches!(
            mode,
            RouteMode::BeamSearch | RouteMode::AStar | RouteMode::Ship
        )
    }

    fn render_options(&mut self, ui: &mut Ui) {
        let Some(mode) = self.ui_state.mode_name.as_ref().cloned() else {
            return;
        };
        self.set_default_mode();
        if self.has_options(&mode) {
            ui.group(|ui| {
                ui.label("Options");
                match mode {
                    RouteMode::BeamSearch => self.options_beam(ui),
                    RouteMode::AStar => self.options_astar(ui),
                    RouteMode::Ship => self.options_ship(ui),
                    _ => (),
                };
            });
        }
    }

    fn mode_help(mode: &RouteMode) -> String {
        match mode {
            RouteMode::BeamSearch => indoc! {"
                Beam-search is a variation of Breadth-First search which prunes the search space by exploring the most promising states first
            "}.to_string(),
            other => formatdoc!("TODO: {other:?}"),
        }
    }

    fn render_modes(&mut self, ui: &mut Ui) {
        let selected = &mut self.ui_state.mode_name;
        egui::ComboBox::from_label("Mode")
            .width(200.0)
            .selected_text(
                selected.map_or_else(
                    || "<None>".to_string(),
                    |name| name.to_string(),
                ),
            )
            .show_ui(ui, |ui| {
                ui.selectable_value(selected, None, "<None>");
                for value in RouteMode::iter() {
                    ui.selectable_value(
                        selected,
                        Some(value),
                        value.to_string(),
                    )
                    .on_hover_text(Self::mode_help(&value));
                }
            });
        self.render_options(ui);
    }

    fn menu_bar(&mut self, ui: &mut Ui, _frame: &eframe::Frame) {
        egui::menu::bar(ui, |ui| {
            ui.menu_button("File", |ui| {
                ui.toggle_value(&mut self.ui_state.show_logs, "Show Logs");
                if ui.button("Exit").clicked() {
                    std::process::exit(0);
                }
            })
        });
    }

    fn run(&mut self, ui: &mut Ui) {
        self.ui_state.last_error = None;
        let Some(mode) = self.ui_state.mode_config.as_ref() else {
            self.ui_state.last_error = Some(anyhow!("No Mode Configured!"));
            return;
        };
        info!("{mode}");
        if let Some(path) = self.ui_state.path.as_ref() {
            let path = path.clone();
            self.exec(|router| {
                router
                    .set_path(path)
                    .map(|_| RouterReturn::None)
                    .map_err(|e| e.into())
            });
        };
        self.exec(|router| Ok(RouterReturn::None));
        // dbg!(&self.ui_state);
    }

    fn router_ui(&mut self, ui: &mut Ui) {
        self.render_modes(ui);
        if ui.button("Run!").clicked() {
            self.run(ui);
        };
    }
}

impl eframe::App for AstronavGUI {
    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        egui::Rgba::TRANSPARENT.to_array() // Make sure we don't paint anything
                                           // behind the rounded corners
    }

    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        // egui::TopBottomPanel::bottom("Logs").resizable(true).show_animated(
        //     ctx,
        //     self.ui_state.show_logs,
        //     |ui| {
        //         let logger = egui_tracing::Logs::new(self.collector.clone());
        //         ui.add(logger);
        //     },
        // );
        egui::CentralPanel::default().show(ctx, |ui| -> AstronavResult<()> {
            self.menu_bar(ui, frame);
            ui.heading(format!("Astronav v{}", env!("CARGO_PKG_VERSION")));
            // let tex = self.ui_state.galaxy_map.get_or_insert_with(|| {
            //     let t_start = Instant::now();
            //     let buffer =
            // galaxy_map::load(r"C:\Users\Earthnuker\astronav\data\stars.bin",
            // 1024).expect("Failed to load heatmap");     println!
            // ("Took: {dt}", dt=t_start.elapsed().human_duration());
            //     ui.ctx().load_texture(
            //         "galaxy_map",
            //         buffer,
            //         Default::default()
            //     )
            // });
            // ui.image(tex.id(), tex.size_vec2());
            self.router_ui(ui);
            while let Ok(res) = self.rx.try_recv() {
                match res {
                    Ok(res) => {
                        info!("RX: {res:?}");
                        self.ui_state.last_error = None;
                    }
                    Err(e) => {
                        error!("RX: {e}");
                        self.ui_state.last_error = Some(e);
                    }
                }
            }
            if let Some(err) = self.ui_state.last_error.as_ref() {
                ui.colored_label(
                    ui.visuals().error_fg_color,
                    format!("Error: {err}"),
                );
            }
            Ok(())
        });
    }
}

pub fn main(router: Router) -> Result<()> {
    let options = NativeOptions {
        // icon_data: Some(
        //     IconData::try_from_png_bytes(&include_bytes!("../../data/icon.
        // png")[..]).unwrap(), ),
        follow_system_theme: true,
        ..Default::default()
    };
    let collector = egui_tracing::EventCollector::default().allowed_targets(
        egui_tracing::tracing::collector::AllowedTargets::Selected(vec![env!(
            "CARGO_PKG_NAME"
        )
        .to_owned()]),
    );
    tracing_subscriber::registry()
        .with(collector.clone())
        .with(
            fmt::layer()
                .event_format(fmt::format().with_ansi(yansi::is_enabled()))
                .with_timer(RelativeTime::default())
                .compact()
                .with_filter(EnvFilter::from_env("ASTRONAV_LOG")),
        )
        .init();
    Ok(eframe::run_native(
        "AstroNav",
        options,
        Box::new(|ctx| Box::new(AstronavGUI::new(ctx, router, collector))),
    )?)
}
