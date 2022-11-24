pub enum RenderAssetType {
	None,
	Texture,
	Mesh,
	SkeletalMesh,
}

#[derive(Debug)]
pub enum ImageFormat {
	Avif,
	Bmp,
	Farbfeld,
	Gif,
	Hdr,
	Ico,
	Jpeg,
	Png,
	Pnm,
	Tga,
	Tiff,
	WebP,
}

impl Into<image::ImageFormat> for ImageFormat {
	fn into(self) -> image::ImageFormat {
		match self {
			ImageFormat::Avif => image::ImageFormat::Avif,
			ImageFormat::Bmp => image::ImageFormat::Bmp,
			ImageFormat::Farbfeld => image::ImageFormat::Farbfeld,
			ImageFormat::Gif => image::ImageFormat::Gif,
			ImageFormat::Hdr => image::ImageFormat::Hdr,
			ImageFormat::Ico => image::ImageFormat::Ico,
			ImageFormat::Jpeg => image::ImageFormat::Jpeg,
			ImageFormat::Png => image::ImageFormat::Png,
			ImageFormat::Pnm => image::ImageFormat::Pnm,
			ImageFormat::Tga => image::ImageFormat::Tga,
			ImageFormat::Tiff => image::ImageFormat::Tiff,
			ImageFormat::WebP => image::ImageFormat::WebP,
		}
	}
}

trait StreamableAsset {
	fn get_render_asset_type(&self) -> RenderAssetType {
		RenderAssetType::None
	}

	// LOD = Level of Detail
	fn calculate_cumulative_lod_size(&self) -> i32;
}
