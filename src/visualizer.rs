use std::{path::PathBuf, thread::JoinHandle, time::Instant};

use crossbeam_channel::{bounded, Receiver, RecvError, Sender, TryRecvError};
use eyre::Result;
use human_repr::HumanDuration;
use pixels::{Pixels, SurfaceTexture};
use rayon::prelude::*;
use tracing::*;
use winit::{
    dpi::LogicalSize,
    event::{Event, VirtualKeyCode},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};
use winit_input_helper::WinitInputHelper;

use crate::{
    common::{StarKind, TreeNode},
    data_loader,
};

struct Visualizer {
    render_thread: JoinHandle<Result<()>>,
    tx: Sender<Vec<TreeNode>>,
}

struct RenderState {

}

impl Visualizer {
    pub(crate) fn new(width: usize, height: usize) -> Result<Self> {
        let (tx, rx) = bounded(4096);
        let render_thread =
            std::thread::spawn(move || Self::render_thread(width, height, rx));
        Ok(Self { render_thread, tx })
    }

    fn send_nodes(&self, nodes: Vec<TreeNode>) -> Result<()> {
        self.tx.send(nodes).map_err(|e| e.into())
    }

    fn render_thread(
        width: usize,
        height: usize,
        rx: Receiver<Vec<TreeNode>>,
    ) -> Result<()> {
        let heatmap = vec![vec![0f64; width]; height];
        let event_loop = EventLoop::new();
        let mut input = WinitInputHelper::new();
        let window = {
            let size = LogicalSize::new(width as f64, height as f64);
            let scaled_size = LogicalSize::new(width as f64, height as f64);
            WindowBuilder::new()
                .with_title("Galaxy Viewer")
                .with_inner_size(scaled_size)
                .with_min_inner_size(size)
                .build(&event_loop)?
        };
        let mut pixels = {
            let window_size = window.inner_size();
            let surface_texture = SurfaceTexture::new(
                window_size.width,
                window_size.height,
                &window,
            );
            Pixels::new(width.try_into()?, height.try_into()?, surface_texture)?
        };
        event_loop.run(move |event, _, control_flow| {
            if let Event::RedrawRequested(_) = event {
                loop {
                    match rx
                        .try_recv()
                        .map(|nodes| Self::render(nodes, &mut pixels))
                    {
                        Ok(Ok(_)) => (),
                        Ok(Err(err)) => {
                            error!("{err}");
                            *control_flow = ControlFlow::Exit;
                            break;
                        }
                        Err(TryRecvError::Empty) => {
                            break;
                        }
                        Err(err) => {
                            error!("{err}");
                            *control_flow = ControlFlow::Exit;
                            break;
                        }
                    }
                }
            }
            if input.update(&event) {
                if input.key_pressed(VirtualKeyCode::Escape)
                    || input.close_requested()
                {
                    *control_flow = ControlFlow::Exit;
                }
                window.request_redraw();
            }
        });
    }

    fn render(nodes: Vec<TreeNode>, px: &mut Pixels) -> Result<()> {
        info!("TODO: Render state points");
        px.render().map_err(|e| e.into())
    }
}
