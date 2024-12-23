use std::{
    fmt::Display,
    path::PathBuf,
    thread::{self, JoinHandle},
};

use crossbeam_channel::{unbounded, Receiver, Sender};
use enum_iterator::Sequence;
use iced::{
    alignment::{Horizontal, Vertical},
    color, executor,
    mouse::Cursor,
    theme::Palette,
    widget::{
        button,
        canvas::{Geometry, Program},
        pick_list, progress_bar, row, scrollable, text, text_input, tooltip,
        Canvas, Column,
    },
    window::Level,
    Application, Command, Element, Length, Point, Rectangle, Renderer,
    Settings, Theme,
};
use itertools::Itertools;
use tracing::{error, info, warn};

use crate::{
    common::{AstronavError, AstronavResult, BeamWidth, SysEntry, System},
    event::{Event, ProcessState, RouteState},
    route::{ModeConfig, Router},
    ship::{NamedShip, Ship},
};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Sequence)]
enum RouteMode {
    Beam,
    DepthFirst,
    IncrementalBroadening,
    BeamStack,
    AStar,
    Dijkstra,
    Ship,
}

// TODO: migrate to Iced 0.10

impl Display for RouteMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Beam => write!(f, "Beam search"),
            Self::DepthFirst => write!(f, "Depth-first search"),
            Self::IncrementalBroadening => {
                write!(f, "Incrementally-broadening beam search")
            }
            Self::BeamStack => write!(f, "Beam-stack search"),
            Self::AStar => write!(f, "A* search"),
            Self::Dijkstra => write!(f, "Dijkstra shortest path"),
            Self::Ship => write!(f, "Ship simulation"),
        }
    }
}

impl BeamMode {
    fn config_view<'a>(&self, router: &'a RouterUi) -> Element<'a, Message> {
        let mut rows = Vec::new();
        match self {
            other => rows.push(text(format!("TODO: {other:?}")).into()),
        };
        Column::with_children(rows).spacing(10).into()
    }

    fn picker(router: &RouterUi) -> Vec<Element<'_, Message>> {
        let mut rows = Vec::new();
        let beam_modes = enum_iterator::all::<Self>().collect_vec();
        let beam_mode = router.mode.as_ref().and_then(|m| m.beam_mode);
        rows.push(
            pick_list(beam_modes, beam_mode, |mode| {
                Message::SetValue(SetValue::BeamMode(mode))
            })
            .into(),
        );
        if let Some(beam_mode) = beam_mode {
            rows.push(beam_mode.config_view(router));
        }
        rows
    }
}

// TODO: hide UI when route computation is running

struct FSDChart(Ship);

impl FSDChart {
    fn from_ship(ship: &NamedShip) -> Self {
        Self(ship.get_inner())
    }
}

impl Program<Message> for FSDChart {
    type State = ();

    fn draw(
        &self,
        _state: &Self::State,
        renderer: &Renderer,
        theme: &Theme,
        bounds: Rectangle,
        cursor: Cursor,
    ) -> Vec<Geometry> {
        use iced::widget::canvas::*;
        let col = color!(0xff_7f_00);
        let grid_cols =
            (color!(0x40_40_40), color!(0x80_80_80), color!(0xf0_f0_f0));
        let num_samples: usize = 1_000;
        let size = bounds.size();
        let width = size.width;
        let height = size.height;
        let x_scale = width / self.0.fuel_capacity; // px/ton
        let y_scale = height / self.0.max_range(); // px/Ly
        let mut frame = Frame::new(renderer, size);

        for point in 0.. {
            let (stroke_color, stroke_width) = if point % 5 == 0 {
                (grid_cols.1, 1.5)
            } else {
                (grid_cols.0, 1.0)
            };
            let point = point as f32;
            let px = point * x_scale;
            let py = point * y_scale * 0.9;
            if px > width && py > height {
                break;
            }
            frame.stroke(
                &Path::line(Point::new(px, 0.0), Point::new(px, height)),
                Stroke::default()
                    .with_color(stroke_color)
                    .with_width(stroke_width),
            );
            frame.stroke(
                &Path::line(
                    Point::new(0.0, height - py),
                    Point::new(width, height - py),
                ),
                Stroke::default()
                    .with_color(stroke_color)
                    .with_width(stroke_width),
            );
        }
        if let Some(pos) = cursor.position_in(bounds) {
            let fuel_amount = pos.x / x_scale;
            let range = self.0.jump_range(fuel_amount, true);
            let y = range / self.0.max_range();
            let pos = Point::new(pos.x, y.mul_add(-height * 0.9, height));
            frame.fill(&Path::circle(pos, 2.0), col);
            frame.fill_text({
                Text {
                    content: format!("{range:.02} Ly @ {fuel_amount:.02} t"),
                    position: pos,
                    color: theme.palette().text,
                    vertical_alignment: Vertical::Bottom,
                    horizontal_alignment: Horizontal::Left,
                    ..Default::default()
                }
            });
        }
        frame.stroke(
            &Path::new(|b| {
                b.move_to(Point::new(0.0, height));
                for p in 0..=num_samples {
                    let x = (p as f32) / (num_samples as f32);
                    let y = self.0.jump_range(x * self.0.fuel_capacity, true)
                        / self.0.max_range();
                    let p =
                        Point::new(x * width, y.mul_add(-height * 0.9, height));
                    b.line_to(p);
                    b.move_to(p);
                }
            }),
            Stroke::default().with_color(col).with_width(1.0),
        );

        let edges = [
            ((0.0, 0.0), (0.0, height)),
            ((width, 0.0), (width, height)),
            ((0.0, 0.0), (width, 0.0)),
            ((0.0, height), (width, height)),
        ];
        for (p1, p2) in edges {
            frame.stroke(
                &Path::line(Point::new(p1.0, p1.1), Point::new(p2.0, p2.1)),
                Stroke::default().with_color(grid_cols.2).with_width(1.0),
            );
        }
        vec![frame.into_geometry()]
    }
}

impl NamedShip {
    fn range_graph<'r>(&self) -> Element<'r, Message> {
        Canvas::new(FSDChart::from_ship(self)).width(480).height(240).into()
    }

    fn info_view<'r>(&self) -> Vec<Element<'r, Message>> {
        vec![
            text(format!("Ship Type: {}", self.ship_type)).into(),
            text(format!("Ship Identifier: {}", self.ident)).into(),
            text(format!(
                "Ship Name: {}",
                self.ship_name.as_deref().unwrap_or("<No Name>")
            ))
            .into(),
            text(format!(
                "Jump Range with full tank: {}",
                self.jump_range(self.fuel_capacity, true)
            ))
            .into(),
            text(format!(
                "Max Jump Range: {}",
                self.jump_range(self.fsd.max_fuel, true)
            ))
            .into(),
            self.range_graph(),
        ]
    }

    fn picker(router: &RouterUi) -> Vec<Element<'_, Message>> {
        let ships = &router.ships;
        let mut rows = Vec::new();
        let selected_ship =
            router.mode.as_ref().and_then(|m| m.ship.as_ref()).cloned();
        rows.push(
            pick_list(ships, selected_ship.clone(), |ship| {
                Message::SetValue(SetValue::Ship(ship))
            })
            .into(),
        );

        if let Some(ship) = selected_ship.as_ref() {
            for item in ship.info_view() {
                rows.push(item);
            }
        }
        rows
    }
}

impl RouteMode {
    fn config_view<'a>(
        &self,
        router: &'a RouterUi,
    ) -> AstronavResult<Vec<Element<'a, Message>>> {
        let mut rows = Vec::new();
        match self {
            Self::Beam => rows.append(&mut BeamMode::picker(router)),
            Self::Ship => rows.append(&mut NamedShip::picker(router)),
            other => info!("TODO: {other:?}"),
        };
        Ok(rows)
    }

    fn picker(router: &RouterUi) -> AstronavResult<Element<'_, Message>> {
        let mut rows = Vec::new();
        let modes = enum_iterator::all::<Self>().collect_vec();
        let route_mode = router.mode.as_ref().and_then(|m| m.mode);
        rows.push(
            pick_list(modes, route_mode, |mode| {
                Message::SetValue(SetValue::RouteMode(mode))
            })
            .into(),
        );
        if let Some(mode) = route_mode {
            let mut config_view = mode.config_view(router)?;
            rows.append(&mut config_view)
        }
        Ok(Column::with_children(rows).spacing(10).into())
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Sequence)]
enum BeamMode {
    Infinite,
    Absolute,
    Fraction,
    Forward,
}

impl Display for BeamMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Infinite => write!(f, "Infinite"),
            Self::Absolute => write!(f, "Absolute"),
            Self::Fraction => write!(f, "Fraction"),
            Self::Forward => write!(f, "Forward"),
        }
    }
}

#[derive(Debug, Clone)]
enum State {
    Init,
    Ready(Receiver<RouterResponse>),
}

#[derive(Debug, Clone)]
enum SetValue {
    RouteMode(RouteMode),
    BeamMode(BeamMode),
    Ship(NamedShip),
}

#[derive(Debug, Clone)]
enum Message {
    SystemNames(String),
    StarsPathChanged(PathBuf),
    RouterCmd(RouterCmd),
    RouterResponse(RouterResponse),
    CopySystem(System),
    // OpenEDSM(System),
    // OpenSpansh(System),
    ResolveSystems,
    SetValue(SetValue),
    RouterError(String),
}

#[derive(Debug, Clone)]
enum RouterCmd {
    ResolvedSystems(Vec<SysEntry>),
    ComputeRoute { hops: Vec<System>, range: Option<f32>, mode: ModeConfig, mmap_tree: bool },
    SetStarsPath(PathBuf),
    SetChannel(Sender<RouterResponse>),
    Load,
}

#[derive(Debug)]
enum RouterResponse {
    ResolveResult(Vec<(SysEntry, Option<System>)>),
    ComputedRoute(Vec<System>),
    Event(Event),
    Error(AstronavError),
}

impl Clone for RouterResponse {
    fn clone(&self) -> Self {
        match self {
            Self::ResolveResult(res) => Self::ResolveResult(res.clone()),
            Self::ComputedRoute(route) => Self::ComputedRoute(route.clone()),
            Self::Event(ev) => Self::Event(ev.clone()),
            Self::Error(err) => panic!("can't clone {err:?}"),
        }
    }
}

fn router_hanlder(router: Router, rx: Receiver<RouterCmd>) {
    let mut router = router;
    let mut cmd_queue = vec![];
    let tx = loop {
        if let Ok(cmd) = rx.recv() {
            match cmd {
                RouterCmd::SetChannel(ch) => {
                    let callback_ch = ch.clone();
                    router.set_callback(Box::new(
                        move |router: &Router, ev: &Event| {
                            callback_ch
                                .send(RouterResponse::Event(ev.clone()))?;
                            Ok(())
                        },
                    ));
                    break ch;
                }
                other => {
                    cmd_queue.push(other);
                }
            }
        }
    };
    for cmd in cmd_queue.into_iter().chain(rx.into_iter()) {
        info!("CMD: {cmd:?}");
        let send_result = match cmd {
            RouterCmd::SetChannel(_) => unreachable!(),
            RouterCmd::ResolvedSystems(systems) => {
                match router.resolve(&systems) {
                    Ok(res) => {
                        let res =
                            systems.into_iter().zip(res.into_iter()).collect();
                        Some(tx.send(RouterResponse::ResolveResult(res)))
                    }
                    Err(err) => Some(tx.send(RouterResponse::Error(err))),
                }
            }
            RouterCmd::ComputeRoute { hops, range, mode, mmap_tree } => {
                let hops: Vec<u32> = hops.into_iter().map(|s| s.id).collect();
                match router.compute_route(&hops, range, 0.0, mode, mmap_tree) {
                    Ok((_, route)) => {
                        Some(tx.send(RouterResponse::ComputedRoute(route)))
                    }
                    Err(err) => Some(tx.send(RouterResponse::Error(err))),
                }
            }
            RouterCmd::SetStarsPath(path) => {
                if let Err(err) = router.set_path(&path) {
                    Some(tx.send(RouterResponse::Error(err)))
                } else {
                    None
                }
            }
            RouterCmd::Load => {
                if let Err(err) = router.load(&[], 0.0, true) {
                    Some(tx.send(RouterResponse::Error(err)))
                } else {
                    None
                }
            }
        };
        if let Some(Err(e)) = send_result {
            error!("{e}");
        }
    }
}

#[derive(Default)]
struct ModeState {
    mode: Option<RouteMode>,
    beam_mode: Option<BeamMode>,
    beam_width: Option<BeamWidth>,
    ship: Option<NamedShip>,
}

struct RouterUi {
    router_handle: JoinHandle<()>,
    mode: Option<ModeState>,
    route_state: Option<RouteState>,
    preprocess_state: Option<ProcessState>,
    tx: Sender<RouterCmd>,
    error: Option<AstronavError>,
    title: String,
    stars_path: PathBuf,
    route: Vec<System>,
    resolved_systems: Vec<(SysEntry, Option<System>)>,
    system_names: String,
    ships: Vec<NamedShip>,
    messages: Vec<String>,
}

impl RouterUi {
    fn handle_error(&mut self, err: AstronavError) -> Command<Message> {
        error!("{err}");
        self.error = Some(err);
        Command::none()
    }

    fn handle_message(
        &mut self,
        message: Message,
    ) -> AstronavResult<Command<Message>> {
        info!("MSG: {message:?}");
        match message {
            Message::CopySystem(sys) => {
                return Ok(iced::clipboard::write(sys.name));
            }
            Message::SystemNames(names) => {
                self.system_names = names;
            }
            Message::ResolveSystems => {
                let mut systems: Vec<SysEntry> = vec![];
                for name in self.system_names.split(',') {
                    match name.trim().parse() {
                        Ok(entry) => {
                            systems.push(entry);
                        }
                        Err(e) => {
                            return Err(AstronavError::Other(e.into()));
                        }
                    }
                }
                self.tx
                    .send(RouterCmd::ResolvedSystems(systems))
                    .map_err(|e| AstronavError::Other(e.into()))?;
            }
            Message::StarsPathChanged(path) => {
                self.stars_path = path.clone();
                self.tx
                    .send(RouterCmd::SetStarsPath(path))
                    .map_err(|e| AstronavError::Other(e.into()))?;
            }
            Message::RouterResponse(resp) => match resp {
                RouterResponse::ResolveResult(res) => {
                    self.resolved_systems = res;
                    let hops = self
                        .resolved_systems
                        .iter()
                        .filter_map(|(_, sys)| sys.as_ref())
                        .cloned()
                        .collect::<Vec<_>>();
                    self.tx
                        .send(RouterCmd::ComputeRoute {
                            hops,
                            range: Some(48.0),
                            mmap_tree: true,
                            mode: ModeConfig::BeamSearch {
                                beam_width: BeamWidth::Absolute(10_000),
                                refuel_mode: None,
                                refuel_primary: false,
                                boost_primary: false,
                                range_limit: f32::INFINITY,
                            },
                        })
                        .map_err(|e| AstronavError::Other(e.into()))?;
                }
                RouterResponse::ComputedRoute(route) => {
                    self.route = route;
                    self.route_state = None;
                    self.preprocess_state = None;
                }
                RouterResponse::Event(ev) => match ev {
                    Event::SearchState(state) => {
                        self.route_state = Some(state);
                    }
                    Event::ProcessState(state) => {
                        self.preprocess_state = Some(state);
                    }
                    Event::Message(msg) => {
                        self.messages.push(msg);
                    }
                },
                RouterResponse::Error(err) => self.error = Some(err),
            },
            Message::RouterCmd(cmd) => {
                self.tx.send(cmd).map_err(|e| AstronavError::Other(e.into()))?
            }
            Message::SetValue(value) => self.set_value(value),
            other => {
                info!("TODO: {other:?}");
            }
        };
        Ok(Command::none())
    }

    fn set_value(&mut self, value: SetValue) {
        let state = self.mode.get_or_insert_with(Default::default);
        match value {
            SetValue::RouteMode(mode) => {
                state.mode = Some(mode);
            }
            SetValue::BeamMode(mode) => {
                state.beam_mode = Some(mode);
            }
            SetValue::Ship(ship) => state.ship = Some(ship),
        }
    }
}

/*
C:\Users\Earthnuker\astronav\data\stars
Colonia, 0/0/0, #81973396946, :10
*/

impl Application for RouterUi {
    type Message = Message;
    type Executor = executor::Default;
    type Flags = Router;
    type Theme = Theme;

    fn new(router: Router) -> (Self, Command<Message>) {
        let (tx, rx) = unbounded();
        let stars_path =
            PathBuf::from(r"C:\Users\Earthnuker\astronav\data\stars");
        let router_handle = thread::spawn(move || router_hanlder(router, rx));
        tx.send(RouterCmd::SetStarsPath(stars_path.clone()))
            .unwrap_or_else(|_| unreachable!());
        let ships = match Ship::new_from_journal() {
            Ok(ships) => ships,
            Err(err) => {
                warn!("Failed to get ship info: {err}");
                vec![]
            }
        };
        (
            Self {
                router_handle,
                tx,
                error: None,
                route_state: None,
                preprocess_state: None,
                mode: None,
                stars_path,
                system_names: "Sol, Colonia, Beagle Point, Sol".to_owned(),
                resolved_systems: Default::default(),
                route: Default::default(),
                title: String::from("Astronav GUI"),
                ships,
                messages: Vec::new(),
            },
            Command::none(),
        )
    }

    fn title(&self) -> String {
        let route_state =
            self.route_state.as_ref().map(|state| format!("{state}"));
        let preprocess_state =
            self.preprocess_state.as_ref().map(|state| format!("{state}"));
        route_state.or(preprocess_state).unwrap_or_else(|| self.title.clone())
    }

    fn theme(&self) -> iced::Theme {
        iced::theme::Theme::custom("Astronav".to_owned(),Palette {
            background: color!(0x1e_1e_1e, 0.5),
            text: color!(0xc8_c8_c8),
            primary: color!(0xaa_4c_00),
            success: color!(0x00_b4_00),
            danger: color!(0xb4_00_00),
        })
    }

    fn update(&mut self, message: Self::Message) -> Command<Self::Message> {
        match self.handle_message(message) {
            Ok(cmd) => cmd,
            Err(err) => self.handle_error(err),
        }
    }

    fn subscription(&self) -> iced::Subscription<Self::Message> {
        let id = std::any::TypeId::of::<Router>();
        iced::subscription::unfold(id, State::Init, |state| async move {
            match state {
                State::Init => {
                    let (tx, rx) = unbounded();
                    (
                        Message::RouterCmd(RouterCmd::SetChannel(tx)),
                        State::Ready(rx),
                    )
                }
                State::Ready(rx) => match rx.recv() {
                    Ok(res) => (Message::RouterResponse(res), State::Ready(rx)),
                    Err(err) => (
                        Message::RouterError(format!("{err}")),
                        State::Ready(rx),
                    ),
                },
            }
        })
    }

    fn view(&self) -> Element<'_, Self::Message> {
        let mut systems = vec![];
        let mode_config = match RouteMode::picker(self) {
            Ok(cfg) => cfg,
            Err(e) => text(format!("Config error: {e}")).into(),
        };
        if !self.resolved_systems.is_empty() {
            systems = self
                .resolved_systems
                .iter()
                .map(|(sys, result)| {
                    result.as_ref().map_or_else(
                        || text(format!("{sys} => <NOT_FOUND>")).into(),
                        |res| text(format!("{sys} => {res}")).into(),
                    )
                })
                .collect();
        };
        let mut route = vec![];
        if !self.route.is_empty() {
            route = self
                .route
                .iter()
                .map(|sys| {
                    row![
                        text(format!("{sys}")),
                        button("Copy")
                            .padding(0)
                            .width(Length::Shrink)
                            .height(Length::Shrink)
                            .on_press(Message::CopySystem(sys.clone()))
                    ]
                    .into()
                })
                .collect();
        };
        let mut rows: Vec<Element<'_, Self::Message>> = vec![
            text("Settings:").size(30).into(),
            row![
                text_input(
                    "Stars Path",
                    &format!("{}", self.stars_path.display())
                )
                .on_input(|s| Message::StarsPathChanged(PathBuf::from(s))),
                button("Load").on_press(Message::RouterCmd(RouterCmd::Load))
            ]
            .spacing(10)
            .into(),
            row![
                text_input("System names", &self.system_names,)
                    .on_input(Message::SystemNames),
                button("Resolve").on_press(Message::ResolveSystems),
            ]
            .spacing(10)
            .into(),
            mode_config,
        ];
        if let Some(state) = self.route_state.as_ref() {
            rows.push(
                progress_bar(0.0..=100.0, state.prc_done).height(10).into(),
            );
            rows.push(text(format!("{state}")).size(20).into());
        }
        rows.push(
            tooltip(
                "Test content",
                "Test tip",
                tooltip::Position::FollowCursor,
            )
            .into(),
        );
        if !systems.is_empty() {
            rows.push(text("Resolved systems:").size(30).into());
            rows.push(scrollable(Column::with_children(systems)).into());
        }
        if !route.is_empty() {
            rows.push(text("Computed route:").size(30).into());
            rows.push(scrollable(Column::with_children(route)).into());
        }
        Column::with_children(rows).spacing(10).into()
    }
}

pub fn main(router: Router) -> iced::Result {
    RouterUi::run(Settings {
        antialiasing: true,
        window: iced::window::Settings {
            transparent: false,
            decorations: true,
            level: Level::AlwaysOnTop,
            ..Default::default()
        },
        flags: router,
        ..Default::default()
    })
}
