use super::{Camera, CameraType, Object3D};
use web_gpu_core::size::Size;

#[derive(Debug, Clone, Copy, PartialEq)]
struct View {
    enabled: bool,
    full_width: f32,
    full_height: f32,
    offset_x: f32,
    offset_y: f32,
    width: f32,
    height: f32,
}

impl Default for View {
    fn default() -> Self {
        Self {
            enabled: true,
            full_width: 1.0,
            full_height: 1.0,
            offset_x: 0.0,
            offset_y: 0.0,
            width: 1.0,
            height: 1.0,
        }
    }
}

/// PerspectiveCamera
///
///
///
///
#[derive(Debug, Clone, PartialEq)]
pub struct PerspectiveCamera {
    eye: glam::Vec3A,
    target: glam::Vec3A,
    up: glam::Vec3A,
    projection_mat: glam::Mat4,
    view: Option<View>,

    fov: f32, // field of view
    aspect_ratio: f32,
    z_near: f32,
    z_far: f32,
    zoom_factor: f32,
    film_offset: f32,
    film_gauge: f32,
}

impl PerspectiveCamera {
    pub fn new(width: f32, height: f32) -> Self {
        Self {
            projection_mat: glam::Mat4::default(),
            fov: 45.0_f32,
            aspect_ratio: width / height,
            z_near: 0.1,
            z_far: 100.0,
            eye: glam::vec3a(0.0, 0.0, 2.0),
            target: glam::vec3a(0.0, 0.0, -1.0),
            up: glam::Vec3A::Y,
            zoom_factor: 1.0,
            film_gauge: 35.0,
            film_offset: 0.0,
            view: None,
        }
    }
}

impl PerspectiveCamera {
    pub fn set_size(&mut self, size: impl Into<Size>) {
        let size: Size = size.into();
        self.aspect_ratio = (size.width / size.height) as f32;
    }
}

impl PerspectiveCamera {
    fn get_film_width(&self) -> f32 {
        self.aspect_ratio.min(1.0) * self.film_gauge
    }

    fn get_film_hight(&self) -> f32 {
        self.aspect_ratio.max(1.0) * self.film_gauge
    }

    fn get_effective_fov(&self) -> f32 {
        (((self.fov.to_radians() * 0.5).tan() / self.zoom_factor).atan() * 2.0).to_degrees()
    }
}

impl Object3D for PerspectiveCamera {}

impl Camera for PerspectiveCamera {
    fn camera_type(&self) -> CameraType {
        CameraType::Perspective
    }

    fn projection_matrix(&self) -> glam::Mat4 {
        glam::Mat4::perspective_rh_gl(self.fov, self.aspect_ratio, self.z_near, self.z_far)
    }

    fn view_matrix(&self) -> glam::Mat4 {
        // let center = glam::Vec3::from(self.eye + self.target);
        glam::Mat4::look_at_rh(self.eye.into(), self.target.into(), self.up.into())
    }

    fn zoom_factor(&self) -> f32 {
        self.zoom_factor
    }

    fn set_zoom_factor(&mut self, zoom_factor: f32) {
        self.zoom_factor = zoom_factor;
    }

    fn set_position(&mut self, position: glam::Vec3) {
        todo!()
    }

    fn update(&mut self) {
        // let mut top = self.z_near * (self.fov.to_radians() * 0.5).tan() / self.zoom;
        // let mut height = 2.0 * top;
        // let mut width = self.aspect_ratio * height;
        // let mut left = -0.5 * width;

        // self.view.and_then(|view| {
        //     if !view.enabled {
        //         return Some(());
        //     }

        //     let full_width = view.full_width;
        //     let full_height = view.full_height;

        //     left += view.offset_x * width / full_width;
        //     top -= view.offset_y * height / full_height;
        //     width *= view.width / full_width;
        //     height *= view.height / full_height;

        //     Some(())
        // });

        // let skew = self.film_offset;
        // let left = match skew != 0.0 {
        //     true => left + ((self.z_near * skew) / self.get_film_width()),
        //     false => left,
        // };

        // let v2 = make_perspective(
        //     left,
        //     left + width,
        //     top,
        //     top - height,
        //     self.z_near,
        //     self.z_far,
        // );

        let v1 = glam::Mat4::perspective_rh_gl(
            self.fov.to_radians(),
            self.aspect_ratio,
            self.z_near,
            self.z_far,
        );
    }
}

fn make_perspective(
    left: f32,
    right: f32,
    top: f32,
    bottom: f32,
    near: f32,
    far: f32,
) -> glam::Mat4 {
    let x = 2.0 * near / (right - left);
    let y = 2.0 * near / (top - bottom);

    let a = (right + left) / (right - left);
    let b = (top + bottom) / (top - bottom);
    let c = -(far + near) / (far - near);
    let d = -2.0 * far * near / (far - near);

    glam::Mat4::from_cols(
        glam::Vec4::new(x, 0.0, 0.0, 0.0),
        glam::Vec4::new(0.0, y, 0.0, 0.0),
        glam::Vec4::new(a, b, c, -1.0),
        glam::Vec4::new(0.0, 0.0, d, 1.0),
    )
}

#[cfg(test)]
mod test {
    use web_gpu_core::size::Size;

    use super::PerspectiveCamera;

    #[test]
    fn test_set_size() {
        let mut p = PerspectiveCamera::new(10.0, 1.0);
        p.set_size(Size::new(10.0, 10.0));
    }
}
