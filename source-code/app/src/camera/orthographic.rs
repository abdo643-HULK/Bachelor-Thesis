use super::{Camera, Object3D, Projection};

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct OrthographicCamera {
    left: f32,
    right: f32,
    top: f32,
    bottom: f32,
    z_near: f32,
    z_far: f32,
    zoom_factor: f32,
}

impl OrthographicCamera {
    pub fn new(width: f32, height: f32) -> Self {
        // Self {
        //     z_near: 0.1,
        //     z_far: 100.0,
        //     bottom:
        // }
        Self::default()
    }
}

impl OrthographicCamera {
    fn update_view_proj(&mut self) {
        glam::DAffine3;
        glam::Affine3A;
    }
}

impl Object3D for OrthographicCamera {}

impl Camera for OrthographicCamera {
    fn new(asptect_ratio: super::Size) -> Self {
        todo!()
    }

    fn camera_type(&self) -> Projection {
        Projection::Orthographic
    }

    fn projection_matrix(&self) -> glam::Mat4 {
        // glam::Mat4::perspective_rh_gl(self.fov, self.aspect_ratio, self.z_near, self.z_far)
        glam::Mat4::default()
    }

    fn view_matrix(&self) -> glam::Mat4 {
        // let center = glam::Vec3::from(self.eye + self.target);
        // glam::Mat4::look_at_rh(self.eye.into(), center, self.up.into())
        glam::Mat4::default()
    }

    fn zoom_factor(&self) -> f32 {
        self.zoom_factor
    }

    fn set_zoom_factor(&mut self, zoom: f32) {
        self.zoom_factor = zoom;
    }

    fn set_position(&mut self, positions: glam::Vec3) {
        todo!()
    }

    fn update(&mut self) {}
}

impl Default for OrthographicCamera {
    fn default() -> Self {
        Self {
            left: -1.0,
            right: 1.0,
            bottom: -1.0,
            top: 1.0,
            z_near: 0.1,
            z_far: 2000.0,
            zoom_factor: 1.0,
        }
    }
}
