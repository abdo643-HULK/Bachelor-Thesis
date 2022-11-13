#[derive(Debug, Clone, Copy, Default)]
pub struct Plane {
    normal_d: glam::Vec4,
}

impl Plane {
    #[inline]
    pub const fn new(normal: glam::Vec3, d: f32) -> Self {
        use glam::{vec4, Vec3};

        let Vec3 { x, y, z } = normal;

        Self {
            normal_d: vec4(x, y, z, d),
        }
    }
}

impl Plane {
    #[inline]
    pub fn normal(&self) -> glam::Vec3A {
        glam::Vec3A::from(self.normal_d)
    }

    #[inline]
    pub fn d(&self) -> f32 {
        self.normal_d.w
    }

    #[inline]
    pub fn normal_d(&self) -> glam::Vec4 {
        self.normal_d
    }
}

impl From<glam::Vec4> for Plane {
    fn from(normal_d: glam::Vec4) -> Self {
        use glam::Vec4Swizzles;

        Self {
            normal_d: normal_d * normal_d.xyz().length_recip(),
        }
    }
}
