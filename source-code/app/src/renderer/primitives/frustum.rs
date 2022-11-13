use super::plane::Plane;

#[derive(Clone, Copy, Debug, Default)]
pub struct Frustum {
    planes: [Plane; 6],
}
