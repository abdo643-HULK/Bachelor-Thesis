pub mod first;
pub mod orthographic;
pub mod perspective;

pub enum CameraType {
    Perspective,
    Orthographic,
}

pub trait Object3D {
    fn position(&self) -> &glam::Vec3 {
        &glam::Vec3::ZERO
    }
}

pub struct AspectRatio(pub f64, pub f64);

pub trait Camera: Object3D {
    fn new(asptect_ratio: AspectRatio) -> Self;
    fn up(&self) -> glam::Vec3 {
        glam::Vec3::Y
    }

    fn camera_type(&self) -> CameraType;
    fn zoom_factor(&self) -> f32;
    fn view_matrix(&self) -> glam::Mat4;
    fn projection_matrix(&self) -> glam::Mat4;

    fn inverse_projection_matrix(&self) -> glam::Mat4 {
        self.projection_matrix().inverse()
    }

    fn set_position(&mut self, position: glam::Vec3);
    fn set_zoom_factor(&mut self, zoom_factor: f32);

    fn update(&mut self);
}

pub struct GameCamera<T: Camera> {
    camera: T,
}

impl<T: Camera> GameCamera<T> {
    pub fn set_camera(&mut self, camera: T) {
        self.camera = camera;
    }
}
