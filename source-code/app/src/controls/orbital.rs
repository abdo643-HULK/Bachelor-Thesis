use derivative::Derivative;
use glam::Vec3;
use winit::{dpi::PhysicalPosition, event::ModifiersState};

use crate::camera::{
    self, orthographic::OrthographicCamera, perspective::PerspectiveCamera, AspectRatio, Camera,
    CameraType, Object3D,
};

use std::{f32, fmt::Debug, sync::Arc};

use super::Controls;

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Spherical {
    pub radius: f32,
    pub phi: f32,
    pub theta: f32,
}

impl Spherical {
    pub fn new() -> Self {
        Self {
            radius: 1.0,
            phi: 0.0,   // polar angle
            theta: 0.0, // azimuthal angle
        }
    }
}

pub enum SphericalSetter {
    Vec(glam::Vec3),
    Values(f32, f32, f32),
    CartesianCoords(f32, f32, f32),
}

impl Spherical {
    pub fn set(&mut self, input: SphericalSetter) -> &mut Self {
        let (radius, phi, theta) = match input {
            SphericalSetter::Vec(input) => {
                let Spherical { radius, phi, theta } = Spherical::from(input);
                (radius, phi, theta)
            }
            SphericalSetter::CartesianCoords(x, y, z) => {
                let Spherical { radius, phi, theta } = Spherical::from(glam::vec3(x, y, z));
                (radius, phi, theta)
            }
            SphericalSetter::Values(radius, phi, theta) => (radius, phi, theta),
        };

        self.radius = radius;
        self.phi = phi;
        self.theta = theta;

        self
    }

    pub fn make_safe(&mut self) -> &mut Self {
        let eps = 0.000001_f32;
        self.phi = eps.max((f32::consts::PI - eps).min(self.phi));

        self
    }
}

impl From<glam::Vec3> for Spherical {
    fn from(input: glam::Vec3) -> Self {
        let radius = (input.x * input.x + input.y * input.y + input.z * input.z).sqrt();

        let (theta, phi) = if radius == 0.0 {
            (0.0, 0.0)
        } else {
            let theta = input.x.atan2(input.z);
            let phi = (input.y / radius).clamp(-1.0, 1.0).acos();
            (theta, phi)
        };

        Self { radius, phi, theta }
    }
}

///
///
///
/// OrbitControls
///
///
///
#[derive(Debug, Clone, PartialEq)]
pub enum AutoRotation {
    Disable,
    Enable(f32),
}

impl AutoRotation {
    fn is_enabled(&self) -> bool {
        matches!(self, Self::Enable(_))
    }

    fn is_disabled(&self) -> bool {
        matches!(self, Self::Disable)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Pan {
    Disable,
    Enable(f32),
}

impl Pan {
    fn is_enabled(&self) -> bool {
        matches!(self, Self::Enable(_))
    }

    fn is_disabled(&self) -> bool {
        matches!(self, Self::Disable)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Zoom {
    Disable,
    Enable(f32),
}

impl Zoom {
    fn is_enabled(&self) -> bool {
        matches!(self, Self::Enable(_))
    }

    fn is_disabled(&self) -> bool {
        matches!(self, Self::Disable)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Damping {
    Disable,
    Enable(f32),
}

impl Damping {
    fn is_enabled(&self) -> bool {
        matches!(self, Self::Enable(_))
    }

    fn is_disabled(&self) -> bool {
        matches!(self, Self::Disable)
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
enum ControllerState {
    #[default]
    None,
    Rotate,
    Dolly,
    Pan,
    TouchRotate,
    TouchPan,
    TouchDollyPan,
    TouchDollyRotate,
}

// impl Default for ControllerState {
//     fn default() -> Self {
//         ControllerState::None
//     }
// }
#[derive(Derivative)]
// #[derive(Debug, Clone)]
#[derivative(Debug, Clone, PartialEq, Default)]
struct Controller {
    #[derivative(Default(value = "1.0"))]
    scale: f32,
    state: ControllerState,
    pan_offset: glam::Vec3A,

    rotate_start: glam::Vec2,
    rotate_end: glam::Vec2,
    rotate_delta: glam::Vec2,

    pan_start: glam::Vec2,
    pan_end: glam::Vec2,
    pan_delta: glam::Vec2,

    dolly_start: glam::Vec2,
    dolly_end: glam::Vec2,
    dolly_delta: glam::Vec2,
}

// impl Default for Controller {
//     fn default() -> Self {
//         Self {
//             scale: 1.0,
//             dolly_delta: Default::default(),
//             dolly_end: Default::default(),
//             dolly_start: Default::default(),
//             pan_delta: Default::default(),
//             pan_end: Default::default(),
//             pan_offset: Default::default(),
//             pan_start: Default::default(),
//             rotate_delta: Default::default(),
//             rotate_end: Default::default(),
//             rotate_start: Default::default(),
//             state: Default::default(),
//         }
//     }
// }

#[derive(Derivative, Debug, Clone, Copy, PartialEq)]
#[derivative(Default)]
struct State {
    #[derivative(Default(value = " glam::Vec3A::ZERO"))]
    target: glam::Vec3A,
    position: glam::Vec3A,
    zoom_factor: f32,
}

// #[derive(Clone)]
// struct EventHandlers {
//     pan_left: Arc<dyn FnMut(f32, glam::Mat4)>,
// }

// impl Debug for EventHandlers {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         f.debug_struct("EventHandlers")
//             .field("pan_left", &"Box<dyn FnMut(f32, glam::Mat4)")
//             .finish()
//     }
// }

#[derive(Debug, Clone, PartialEq)]
pub enum Cam<T: camera::Camera> {
    Perspective(PerspectiveCamera),
    Orthographic(OrthographicCamera),
    Custom(T),
}

impl<T: camera::Camera> Object3D for Cam<T> {}

impl<T: camera::Camera> camera::Camera for Cam<T> {
    fn camera_type(&self) -> CameraType {
        match self {
            Cam::Perspective(camera) => camera.camera_type(),
            Cam::Orthographic(camera) => camera.camera_type(),
            Cam::Custom(camera) => camera.camera_type(),
        }
    }

    fn zoom_factor(&self) -> f32 {
        match self {
            Cam::Perspective(camera) => camera.zoom_factor(),
            Cam::Orthographic(camera) => camera.zoom_factor(),
            Cam::Custom(camera) => camera.zoom_factor(),
        }
    }

    fn view_matrix(&self) -> glam::Mat4 {
        match self {
            Cam::Perspective(camera) => camera.view_matrix(),
            Cam::Orthographic(camera) => camera.view_matrix(),
            Cam::Custom(camera) => camera.view_matrix(),
        }
    }

    fn projection_matrix(&self) -> glam::Mat4 {
        match self {
            Cam::Perspective(camera) => camera.projection_matrix(),
            Cam::Orthographic(camera) => camera.projection_matrix(),
            Cam::Custom(camera) => camera.projection_matrix(),
        }
    }

    fn set_position(&mut self, position: glam::Vec3) {
        match self {
            Cam::Perspective(camera) => camera.set_position(position),
            Cam::Orthographic(camera) => camera.set_position(position),
            Cam::Custom(camera) => camera.set_position(position),
        }
    }

    fn set_zoom_factor(&mut self, zoom_factor: f32) {
        match self {
            Cam::Perspective(camera) => camera.set_zoom_factor(zoom_factor),
            Cam::Orthographic(camera) => camera.set_zoom_factor(zoom_factor),
            Cam::Custom(camera) => camera.set_zoom_factor(zoom_factor),
        }
    }

    fn update(&mut self) {
        match self {
            Cam::Perspective(camera) => camera.update(),
            Cam::Orthographic(camera) => camera.update(),
            Cam::Custom(camera) => camera.update(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct Bound<T> {
    min: T,
    max: T,
}

#[derive(Derivative, Debug, Clone, PartialEq)]
#[derivative(Default)]
pub struct ControlConfig {
    /// How far you can dolly in and out ( PerspectiveCamera only )
    #[derivative(Default(value = "Bound {
        min: 0.0,
        max: f32::INFINITY,
    }"))]
    distance: Bound<f32>,
    /// How far you can zoom in and out ( OrthographicCamera only )
    #[derivative(Default(value = "Bound {
        min: 0.0,
        max: f32::INFINITY,
    }"))]
    zoom: Bound<f32>,
    /// How far you can orbit vertically, upper and lower limits.
    /// Range is 0 to Math.PI radians.
    #[derivative(Default(value = "Bound {
        min: 0.0,
        max: f32::consts::PI,
    }"))]
    polar_angle: Bound<f32>, // radians
    /// How far you can orbit horizontally, upper and lower limits.
    /// If set, the interval [ min, max ]  must be a sub-interval of the interval [ - 2 PI, 2 PI ]. with ( max - min < 2 PI )
    #[derivative(Default(value = "Bound {
        min: f32::NEG_INFINITY,
        max: f32::INFINITY,
    }"))]
    azimuth_angle: Bound<f32>, // radians
}

#[derive(Debug, Clone, PartialEq)]
pub struct OrbitControls<T: camera::Camera = PerspectiveCamera> {
    #[cfg(feature = "specialization")]
    pub camera: T,
    #[cfg(not(feature = "specialization"))]
    pub camera: Cam<T>,

    target: glam::Vec3A,

    bounds: ControlConfig,

    scale: f32,

    zoom: Zoom,
    auto_rotate: AutoRotation,
    panning: Pan,
    damping: Damping,

    spherical: Spherical,
    spherical_delta: Spherical,

    last_state: State,
    controller: Controller,
    input_state: InputState,
}

impl OrbitControls {
    const EPS: f32 = 0.000001;
    const TWO_PI: f32 = f32::consts::PI * 2.0;
}

impl<T: camera::Camera> OrbitControls<T> {
    fn new(camera: Cam<T>) -> Self {
        Self {
            camera,
            ..Default::default()
        }
    }
}

impl<T: camera::Camera> OrbitControls<T> {
    pub fn polar_angle(&self) -> f32 {
        self.spherical.phi
    }

    pub fn azimuth_angle(&self) -> f32 {
        self.spherical.theta
    }

    pub fn distance(&self) -> f32 {
        glam::Vec3A::from(*self.camera.position()).distance(self.target)
    }

    pub fn auto_rotation_angle(&self, rotation_speed: f32) -> f32 {
        ((2.0 * f32::consts::PI) / 60.0 / 60.0) * rotation_speed
    }

    pub fn zoom_scale(&self, zoom_speed: f32) -> f32 {
        0.95_f32.powf(zoom_speed)
    }
}

impl<T: camera::Camera> OrbitControls<T> {
    fn rotate_up(&mut self, angle: f32) {
        self.spherical_delta.phi -= angle;
    }

    fn rotate_left(&mut self, angle: f32) {
        self.spherical_delta.theta -= angle;
    }
}

struct WheelEvent {
    device_id: winit::event::DeviceId,
    delta: winit::event::MouseScrollDelta,
    phase: winit::event::TouchPhase,
}

// #[derive(Debug, Clone, Copy)]
#[derive(Debug, Clone, Copy, PartialEq)]
struct MouseEvent {
    button: winit::event::MouseButton,
    modifiers: winit::event::ModifiersState,
    position: winit::dpi::PhysicalPosition<f64>,
}

// #[derive(Debug, Clone, Copy, Default)]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
struct TouchEvent {
    modifiers: winit::event::ModifiersState,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
enum Pointer {
    #[default]
    None,
    Mouse(MouseEvent),
    Touch(TouchEvent),
    Pen,
}

#[derive(Debug, Clone, PartialEq, Copy)]
struct PointerEvent {
    pointer_id: winit::event::DeviceId,
    input: Pointer,
    ctrl: bool,
    shift: bool,
    alt: bool,
    meta: bool,
}

impl Event for PointerEvent {}

pub struct KeyboardEvent {}

pub enum MouseAction {
    Dolly,
    Rotate,
    Pan,
}
trait Event: Send + Sync {}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default, Hash)]
enum EvenState {
    #[default]
    None,
    PointerDown,
    PointerMove,
    PointerUp,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct InputState {
    // event: dyn Event,
    event: Option<PointerEvent>,
    current_state: EvenState,
    modifiers: ModifiersState,
}

trait EventHandler {
    fn on_key_down(&mut self, event: KeyboardEvent);
    fn on_mouse_wheel(&mut self, event: WheelEvent);
    fn on_pointer_down(&mut self, event: PointerEvent);
    fn on_pointer_up(&mut self, event: PointerEvent);
    fn on_pointer_move(&mut self, event: PointerEvent);
    fn on_pointer_cancel(&mut self, event: PointerEvent);
}

impl<T: camera::Camera> EventHandler for OrbitControls<T> {
    fn on_key_down(&mut self, event: KeyboardEvent) {}

    fn on_mouse_wheel(&mut self, event: WheelEvent) {
        if self.zoom.is_disabled() || !matches!(self.controller.state, ControllerState::None) {
            return;
        }

        if let Zoom::Enable(zoom) = self.zoom {
            let zoom_scale = self.zoom_scale(zoom);

            // self.dispatchEvent(_startEvent);

            let delta_y = match event.delta {
                winit::event::MouseScrollDelta::LineDelta(x, y) => y as f64,
                winit::event::MouseScrollDelta::PixelDelta(PhysicalPosition { y, .. }) => y,
            };

            match delta_y {
                y if y > 0.0 => self.dolly_out(zoom_scale),
                y if y < 0.0 => self.dolly_in(zoom_scale),
                _ => {}
            };

            // self.dispatchEvent(_endEvent);
        }
    }

    fn on_pointer_down(&mut self, event: PointerEvent) {}

    fn on_pointer_up(&mut self, event: PointerEvent) {
        todo!()
    }

    fn on_pointer_move(&mut self, event: PointerEvent) {
        todo!()
    }

    fn on_pointer_cancel(&mut self, event: PointerEvent) {
        todo!()
    }
}

impl<T: camera::Camera> OrbitControls<T> {
    pub fn on_event(&mut self, event: winit::event::WindowEvent) {
        use winit::event::WindowEvent;

        match event {
            WindowEvent::ModifiersChanged(ev) => {
                self.input_state.modifiers = ev;
            }
            WindowEvent::MouseWheel {
                device_id,
                delta,
                phase,
                ..
            } => self.on_mouse_wheel(WheelEvent {
                delta,
                device_id,
                phase,
            }),
            WindowEvent::Touch(touch) => {
                let event = PointerEvent {
                    pointer_id: touch.device_id,
                    input: Pointer::Touch(TouchEvent {
                        modifiers: self.input_state.modifiers,
                    }),
                    alt: self.input_state.modifiers.alt(),
                    ctrl: self.input_state.modifiers.ctrl(),
                    shift: self.input_state.modifiers.shift(),
                    meta: self.input_state.modifiers.logo(),
                };

                match touch.phase {
                    winit::event::TouchPhase::Started => self.on_pointer_down(event),
                    winit::event::TouchPhase::Moved => self.on_pointer_move(event),
                    winit::event::TouchPhase::Ended => self.on_pointer_up(event),
                    winit::event::TouchPhase::Cancelled => self.on_pointer_cancel(event),
                }
            }
            WindowEvent::MouseInput {
                button,
                device_id,
                state,
                ..
            } => {
                let event = PointerEvent {
                    pointer_id: device_id,
                    input: Pointer::Mouse(MouseEvent {
                        button,
                        modifiers: self.input_state.modifiers,
                        position: Default::default(),
                    }),
                    alt: self.input_state.modifiers.alt(),
                    ctrl: self.input_state.modifiers.ctrl(),
                    shift: self.input_state.modifiers.shift(),
                    meta: self.input_state.modifiers.logo(),
                };

                match state {
                    winit::event::ElementState::Pressed => {
                        self.on_pointer_down(event);
                        self.input_state = InputState {
                            event: Some(event),
                            current_state: EvenState::PointerDown,
                            modifiers: self.input_state.modifiers,
                        }
                    }
                    winit::event::ElementState::Released => {
                        self.on_pointer_up(event);
                        self.input_state = InputState {
                            event: Some(event),
                            current_state: EvenState::PointerUp,
                            modifiers: self.input_state.modifiers,
                        }
                    }
                }
            }
            WindowEvent::CursorMoved {
                device_id,
                position,
                ..
            } => {
                if let Some(event) = self.input_state.event {
                    let event = PointerEvent {
                        pointer_id: device_id,
                        input: match event.input {
                            Pointer::None => todo!(),
                            Pointer::Mouse(mouse) => Pointer::Mouse(MouseEvent {
                                button: mouse.button,
                                modifiers: self.input_state.modifiers,
                                position,
                            }),
                            Pointer::Touch(touch) => todo!(),
                            Pointer::Pen => todo!(),
                        },
                        alt: self.input_state.modifiers.alt(),
                        ctrl: self.input_state.modifiers.ctrl(),
                        shift: self.input_state.modifiers.shift(),
                        meta: self.input_state.modifiers.logo(),
                    };
                    self.on_pointer_move(event)
                }
            }
            WindowEvent::KeyboardInput {
                device_id,
                input,
                is_synthetic,
            } => {
                input.virtual_keycode.and_then(|code| {
                    match code {
                        winit::event::VirtualKeyCode::A => todo!(),
                        winit::event::VirtualKeyCode::D => todo!(),
                        winit::event::VirtualKeyCode::S => todo!(),
                        winit::event::VirtualKeyCode::W => todo!(),
                        winit::event::VirtualKeyCode::Escape => todo!(),

                        winit::event::VirtualKeyCode::Left => todo!(),
                        winit::event::VirtualKeyCode::Up => todo!(),
                        winit::event::VirtualKeyCode::Right => todo!(),
                        winit::event::VirtualKeyCode::Down => todo!(),
                        _ => {}
                    }
                    Some(())
                });
            }
            _ => {}
        }
    }

    fn on_mouse_down(&self, event: MouseEvent) {
        let action = match event.button {
            winit::event::MouseButton::Left => MouseAction::Rotate,
            winit::event::MouseButton::Middle => MouseAction::Dolly,
            winit::event::MouseButton::Right => MouseAction::Pan,
            winit::event::MouseButton::Other(_) => todo!(),
        };
    }
}

impl<T: camera::Camera> OrbitControls<T> {
    fn dolly_in(&mut self, scale: f32) {
        match &mut self.camera {
            Cam::Perspective(_) => self.scale *= scale,
            Cam::Orthographic(camera) => {
                let Bound {
                    min: min_zoom,
                    max: max_zoom,
                } = self.bounds.zoom;

                let zoom = min_zoom.max(max_zoom.min(camera.zoom_factor() / scale));
                camera.set_zoom_factor(zoom);
                camera.update();
                // zoomChanged = true;
            }
            Cam::Custom(_) => self.zoom = Zoom::Disable,
        }
    }

    fn dolly_out(&mut self, scale: f32) {
        match &mut self.camera {
            Cam::Perspective(_) => self.scale /= scale,
            Cam::Orthographic(camera) => {
                let Bound {
                    min: min_zoom,
                    max: max_zoom,
                } = self.bounds.zoom;

                let zoom = min_zoom.max(max_zoom.min(camera.zoom_factor() * scale));
                camera.set_zoom_factor(zoom);
                camera.update();
                // zoomChanged = true;
            }
            Cam::Custom(_) => self.zoom = Zoom::Disable,
        }
    }
}

#[cfg(feature = "specialization")]
impl OrbitControls<PerspectiveCamera> {
    fn dolly_in(&mut self, scale: f32) {
        self.scale *= scale
    }
}

#[cfg(feature = "specialization")]
impl OrbitControls<OrthographicCamera> {
    fn dolly_in(&mut self, scale: f32) {
        let Bound {
            min: min_zoom,
            max: max_zoom,
        } = self.bounds.zoom;

        let zoom = min_zoom.max(max_zoom.min(camera.zoom_factor() / scale));
        self.camera.set_zoom_factor(zoom);
        self.camera.update();
        // zoomChanged = true;
    }
}

impl<T: camera::Camera> Controls for OrbitControls<T> {
    fn save(&mut self) {
        self.last_state.target = self.target; // a copy
        self.last_state.position = (*self.camera.position()).into(); // a copy
        self.last_state.zoom_factor = self.camera.zoom_factor();
    }

    fn reset(&mut self) {
        self.target = self.last_state.target;
        self.camera.set_position(self.last_state.position.into());
        self.camera.set_zoom_factor(self.last_state.zoom_factor);
        self.camera.update();
        // scope.dispatchEvent( _changeEvent );

        self.update();
        self.controller.state = ControllerState::None;
    }

    fn update(&mut self) {
        let quat = glam::Quat::from_rotation_arc(self.camera.up(), glam::Vec3::Y);
        let quat_inverse = quat.inverse();

        let mut last_position = glam::Vec3A::default();
        let mut last_quat = glam::Quat::default();

        let OrbitControls {
            spherical,
            spherical_delta,
            controller,
            ..
        } = self;

        let offset = glam::Vec3A::from(*self.camera.position()) - self.target;
        let offset = quat_inverse * quat * offset;

        spherical.set(SphericalSetter::Vec(offset.into()));

        match self.auto_rotate {
            AutoRotation::Enable(_) if controller.state == ControllerState::None => {
                // rotateLeft( self.auto_rotation_angle() );
            }
            _ => {}
        }

        match self.damping {
            Damping::Disable => {
                spherical.theta += spherical_delta.theta;
                spherical.phi += spherical_delta.phi;
            }
            Damping::Enable(factor) => {
                spherical.theta += spherical_delta.theta * factor;
                spherical.phi += spherical_delta.phi * factor;
            }
        }

        let Bound {
            min: min_polar_angle,
            max: max_polar_angle,
        } = self.bounds.polar_angle;

        // restrict phi to be between desired limits
        spherical.phi = min_polar_angle.max(max_polar_angle.min(spherical.phi));

        spherical.make_safe();

        spherical.radius *= controller.scale;

        let Bound {
            min: min_distance,
            max: max_distance,
        } = self.bounds.distance;

        // restrict radius to be between desired limits
        spherical.radius = min_distance.max(max_distance.min(spherical.radius));

        // move target to panned location
        match self.damping {
            Damping::Disable => self.target += controller.pan_offset,
            Damping::Enable(factor) => self.target += controller.pan_offset * factor,
        }
    }
}

impl<T: camera::Camera> Default for OrbitControls<T> {
    fn default() -> Self {
        let camera = T::new(AspectRatio(1920.0, 1080.0));
        let position = (*camera.position()).into();
        let zoom_factor = camera.zoom_factor();
        let target = glam::Vec3A::ZERO;

        let pan_left = {
            let mut v = glam::Vec3A::default();

            Arc::new(move |distance: f32, object_matrix: glam::Mat4| {
                let col = object_matrix.col(0);
                v.x = col.x;
                v.y = col.y;
                v.z = col.z;

                v = v * -distance;

                // panOffset.add(v);
            })
        };

        let pan_up = {
            let mut v = glam::Vec3A::default();

            Arc::new(
                move |scope: &Self, distance: f32, object_matrix: glam::Mat4| {
                    match scope.panning {
                        Pan::Disable => {
                            let col = object_matrix.col(1);
                            v.x = col.x;
                            v.y = col.y;
                            v.z = col.z;
                        }
                        Pan::Enable(_) => {
                            let col = object_matrix.col(0);
                            let internal_v = glam::vec3(col.x, col.y, col.z);
                            v = scope.camera.up().cross(internal_v).into()
                        }
                    }

                    v = v * distance;
                    // panOffset.add(v);
                },
            )
        };

        Self {
            camera: Cam::Perspective(camera),
            target,

            bounds: ControlConfig::default(),

            zoom: Zoom::Enable(1.0),
            auto_rotate: AutoRotation::Enable(1.0),
            panning: Pan::Enable(1.0),
            damping: Damping::Disable,

            spherical: Spherical::new(),
            spherical_delta: Spherical::new(),

            scale: 1.9,
            last_state: State {
                position,
                zoom_factor,
                ..Default::default()
            },
            controller: Controller::default(),
            input_state: Default::default(),
        }
    }
}

#[cfg(feature = "specialization")]
impl Default for OrbitControls<PerspectiveCamera> {
    fn default() -> Self {
        let camera = PerspectiveCamera::new(1920.0, 1080.0);
        let position = (*camera.position()).into();
        let zoom = camera.zoom_factor();
        let target = glam::Vec3A::ZERO;

        let pan_left = {
            let mut v = glam::Vec3A::default();

            Arc::new(move |distance: f32, object_matrix: glam::Mat4| {
                let col = object_matrix.col(0);
                v.x = col.x;
                v.y = col.y;
                v.z = col.z;

                v = v * -distance;

                // panOffset.add(v);
            })
        };

        let pan_up = {
            let mut v = glam::Vec3A::default();

            Arc::new(
                move |scope: &Self, distance: f32, object_matrix: glam::Mat4| {
                    match scope.panning {
                        Pan::Disable => {
                            let col = object_matrix.col(1);
                            v.x = col.x;
                            v.y = col.y;
                            v.z = col.z;
                        }
                        Pan::Enable(_) => {
                            let col = object_matrix.col(0);
                            let internal_v = glam::vec3(col.x, col.y, col.z);
                            v = scope.camera.up().cross(internal_v).into()
                        }
                    }

                    v = v * distance;
                    // panOffset.add(v);
                },
            )
        };

        fn dolly_in(scope: &mut OrbitControls, scale: f32) {
            let Bound {
                min: min_zoom,
                max: max_zoom,
            } = scope.bounds.zoom;

            let zoom = min_zoom.max(max_zoom.min(scope.camera.zoom_factor() / scale));
            scope.camera.set_zoom_factor(zoom);
            scope.camera.update();
            // zoomChanged = true;
        }

        Self {
            camera: Cam::Perspective(camera),
            target: target.clone(),

            bounds: ControlConfig::default(),

            zoom: Zoom::Enable(1.0),
            auto_rotate: AutoRotation::Enable(1.0),
            panning: Pan::Enable(1.0),
            damping: Damping::Disable,

            spherical: Spherical::new(),
            spherical_delta: Spherical::new(),

            scale: 1.9,
            last_state: State {
                target: glam::Vec3A::ZERO,
                position,
                zoom_factor: zoom,
            },
            controller: Controller::default(),
            // event_handlers: EventHandlers { pan_left },
        }
    }
}

#[cfg(feature = "specialization")]
impl Default for OrbitControls<OrthographicCamera> {
    fn default() -> Self {
        let camera = OrthographicCamera::new(1920.0, 1080.0);
        let position = (*camera.position()).into();
        let zoom = camera.zoom_factor();
        let target = glam::Vec3A::ZERO;

        let pan_left = {
            let mut v = glam::Vec3A::default();

            Arc::new(move |distance: f32, object_matrix: glam::Mat4| {
                let col = object_matrix.col(0);
                v.x = col.x;
                v.y = col.y;
                v.z = col.z;

                v = v * -distance;

                // panOffset.add(v);
            })
        };

        let pan_up = {
            let mut v = glam::Vec3A::default();

            Arc::new(
                move |scope: &Self, distance: f32, object_matrix: glam::Mat4| {
                    match scope.panning {
                        Pan::Disable => {
                            let col = object_matrix.col(1);
                            v.x = col.x;
                            v.y = col.y;
                            v.z = col.z;
                        }
                        Pan::Enable(_) => {
                            let col = object_matrix.col(0);
                            let internal_v = glam::vec3(col.x, col.y, col.z);
                            v = scope.camera.up().cross(internal_v).into()
                        }
                    }

                    v = v * distance;
                    // panOffset.add(v);
                },
            )
        };

        fn dolly_in(scope: &mut OrbitControls, scale: f32) {
            scope.scale *= scale;
        }

        Self {
            camera: Cam::Orthographic(camera),
            target: target.clone(),

            bounds: ControlConfig::default(),

            zoom: Zoom::Enable(1.0),
            auto_rotate: AutoRotation::Enable(1.0),
            panning: Pan::Enable(1.0),
            damping: Damping::Disable,

            spherical: Spherical::new(),
            spherical_delta: Spherical::new(),

            scale: 1.9,
            last_state: State {
                target: glam::Vec3A::ZERO,
                position,
                zoom_factor: zoom,
            },
            controller: Controller::default(),
            // event_handlers: EventHandlers { pan_left },
        }
    }
}

#[cfg(test)]
mod test {
    #[cfg(all(target_arch = "x86_64", target_os = "macos"))]
    mod macos_x86 {
        #[test]
        fn distance_test() {
            let position = glam::Vec3A::new(0.0, 1.0, 0.0);
            let target = glam::Vec3A::new(1.0, 0.0, 0.4);

            let distance = position.distance(target);
            println!("{distance}");

            assert_eq!(distance, 1.469693845669907);
        }
    }
}
