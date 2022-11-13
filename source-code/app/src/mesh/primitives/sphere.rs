enum SphereType {
    HemiSphere(Sphere),
    PrimitiveSphere(Sphere),
}

struct Sphere {
    rings: u32,
    radius: f32,
    height: f32,
}
