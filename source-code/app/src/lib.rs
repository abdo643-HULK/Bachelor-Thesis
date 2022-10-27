#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub fn run() {
    #[cfg(target_arch = "wasm32")]
    {
        use winit::{
            event_loop::EventLoopBuilder,
            platform::web::{EventLoopExtWebSys, WindowBuilderExtWebSys},
            window::WindowBuilder,
        };

        #[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
        enum CustomEvent {
            PointerEvent,
        }

        let event_loop = EventLoopBuilder::<CustomEvent>::with_user_event().build();

        let window = WindowBuilder::new()
            .with_prevent_default(false)
            .build(&event_loop)
            .unwrap();

        event_loop.spawn(|event, win, c| match event {
            winit::event::Event::NewEvents(e) => todo!(),
            winit::event::Event::WindowEvent { window_id, event } => todo!(),
            winit::event::Event::DeviceEvent { device_id, event } => todo!(),
            winit::event::Event::UserEvent(ev) => {}
            winit::event::Event::Suspended => todo!(),
            winit::event::Event::Resumed => todo!(),
            winit::event::Event::MainEventsCleared => todo!(),
            winit::event::Event::RedrawRequested(_) => todo!(),
            winit::event::Event::RedrawEventsCleared => todo!(),
            _ => todo!(),
        })
    }
}
