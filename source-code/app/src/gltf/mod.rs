use std::{error::Error, io::BufReader};

struct Gltf;

impl Gltf {
    pub fn load<R>(reader: R) -> Result<(), dyn Error>
    where
        R: io::Read + io::Seek,
    {
        let gltf = gltf::Gltf::from_reader(reader)?;

        for material in gltf.materials() {
            let normal_texture = material.normal_texture();
            let normal_texture = material.occlusion_texture();
            let normal_texture = material
                .pbr_metallic_roughness()
                .metallic_roughness_texture();
        }

        Ok(())
    }

    fn load_mesh() {}

    fn load_material(mat: &gltf::Material) {
        let label = mat.name();
        let pbr = mat.pbr_metallic_roughness();
    }

    async fn load_textures(gltf: &gltf::Gltf) {
        let textures = gltf.textures().map(Gltf::load_texture);
        let textures = futures::future::join_all(textures).await;
    }

    async fn load_texture(tex: gltf::Texture) {
        let sampler = tex.sampler();
        let name = tex.name();
        let image = tex.source();
        let source = image.source();
        // let reader = BufReader::new(inner);
        // image::io::Reader::new(buffered_reader);
        match source {
            gltf::image::Source::View { view, mime_type } => {
                let start = view.offset();
                let end = view.offset() + view.length();
                view.buffer().index()
            }
            gltf::image::Source::Uri { uri, mime_type } => todo!(),
        }
    }
}

struct GltfMaterial {}

struct ImageBuffer {}

struct Material {}
