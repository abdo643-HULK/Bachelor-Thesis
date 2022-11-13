struct Size<T> {
    width: T,
    height: T,
}

struct Image {
    data: Vec<u8>,
    size: Size,
}

impl From for Image {}
