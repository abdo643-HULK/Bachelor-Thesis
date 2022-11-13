use glam::{Affine3A, Mat4, Quat, Vec3A};

struct Transform {
    pub translation: Vec3A,
    pub rotation: Quat,
    pub scale: Vec3A,
}

/// REGION : Constants

impl Transform {
    pub const IDENTITY: Self = Transform {
        translation: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    };
}

/// REGION : public Methods

impl Transform {
    pub fn rotate(&self) {
        let Self {
            translation,
            rotation,
            scale,
        } = self;

        let affine = Affine3A::from_scale_rotation_translation(scale, rotation, translation);
        Mat4::from_cols_array(affine.to_cols_array());
    }
}

/// REGION: From Trait

impl From<Affine3A> for Transform {
    fn from(source: Affine3A) -> Self {
        let (scale, rotation, translation) = source.to_scale_rotation_translation();

        Transform {
            translation,
            rotation,
            scale,
        }
    }
}
