// #![feature(dropck_eyepatch)]
// #![feature(specialization)]

pub mod app;
// pub mod assets;
// pub mod camera;
// pub mod controls;
// mod gltf;
// mod loader;
// mod mesh;
// pub mod model;
// pub mod renderer;
// pub mod triangle;
// pub mod triangle_simple;
// pub mod vertex;

use app::{App, AppOptions};
// use loader::GLTFLoader;

pub const WINDOW_TITLE: &str = "WebGPU Game";
pub const ASSETS_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../assets");

fn main() -> Result<(), Box<dyn std::error::Error>> {
	env_logger::init();

	let (title, path) = (
		"Armadillo",
		format!("{ASSETS_PATH}/models/armadillo/Armadillo.glb"),
	);
	// let (title, path) = (
	//     "Dragon XYZRGB",
	//     format!("{ASSETS_PATH}/models/dragon-xyzrgb/Dragon_xyzrgb.glb"),
	// );
	let (title, path) = ("Dragon", format!("{ASSETS_PATH}/models/dragon/Dragon.glb"));
	// let (title, path) = ("Cube", format!("{ASSETS_PATH}/models/cube/Cube.glb"));
	// let (title, path) = (
	//     "Charizard",
	//     format!("{ASSETS_PATH}/models/charizard/Charizard.gltf"),
	// );

	// let loader = GLTFLoader::new(path, loader::AnimationLoop::Disable);

	// let mut controls = controls::orbital::OrbitControls::default();
	// controls.object.update();

	// App::init(AppOptions {
	// 	title: Some(title),
	// 	..Default::default()
	// })
	// .add_system(Box::new(loader))
	// .run();

	// triangle::run();
	// triangle_simple::run();

	print!("hi");

	let engine = self::engine::Engine::init();
	engine.run();

	return Ok(());
}

// pub struct Inspector<'a, T: std::fmt::Debug>(&'a T);

// unsafe impl<'a, #[may_dangle] T: std::fmt::Debug> Drop for Inspector<'a, T> {
//     fn drop(&mut self) {
//         println!("Inspector({:#?})", self.0);
//     }
// }
mod learn {
	use std::num::NonZeroU32;

	use wgpu::{
		util::{BufferInitDescriptor, DeviceExt},
		*,
	};
	use winit::dpi::LogicalSize;

	const REQUIRED_FEATURES: Features = Features::DEPTH24PLUS_STENCIL8
		.union(Features::SPIRV_SHADER_PASSTHROUGH)
		.union(Features::PUSH_CONSTANTS);
	const REQUIRED_MOBILE_FEATURES: Features = Features::SHADER_FLOAT16;
	const REQUIRED_DESKTOP_FEATURES: Features = Features::SHADER_PRIMITIVE_INDEX;

	pub trait VulkanStencilFaceStateDefault {
		const VULKAN_DEFAULT: StencilFaceState = StencilFaceState {
			compare: CompareFunction::Never,
			fail_op: StencilOperation::Keep,
			depth_fail_op: StencilOperation::Keep,
			pass_op: StencilOperation::Keep,
		};
	}

	impl VulkanStencilFaceStateDefault for StencilFaceState {}

	#[derive(Debug, Copy, Clone)]
	enum SampleCount {
		Bit1 = 0x00000001,
		Bit2 = 0x00000002,
		Bit4 = 0x00000004,
		Bit8 = 0x00000008,
		Bit16 = 0x00000010,
		Bit32 = 0x00000020,
		Bit64 = 0x00000040,
	}

	type Params<'a> = (&'a Device, &'a Queue, &'a Surface, LogicalSize<u32>);
	async fn init(window: &winit::window::Window) {
		let instance = Instance::new(Backends::all());
		let adapters = instance.enumerate_adapters(Backends::all());
		let default_adapter =
			instance.request_adapter(&RequestAdapterOptions::default()).await.unwrap();
		let (device, queue) = default_adapter
			.request_device(
				&DeviceDescriptor {
					label: todo!(),
					features: todo!(),
					limits: todo!(),
				},
				None,
			)
			.await
			.unwrap();
		let surface = unsafe { instance.create_surface(window) };

		let formats = surface.get_supported_formats(&default_adapter);
		let alpha_modes = surface.get_supported_alpha_modes(&default_adapter);

		let formats = dbg!(formats);

		let surface_config = SurfaceConfiguration {
			usage: todo!(),
			format: todo!(),
			width: window.inner_size().width,
			height: window.inner_size().height,
			present_mode: PresentMode::AutoVsync,
			alpha_mode: todo!(),
		};

		surface.configure(&device, &surface_config);
	}

	// wgpu::Surface = vk::Surface & vk::SwapChain
	// wgpu::Device = vk::Device & vk::CommandPool. It allocates commandbuffers
	fn texture_render_pass(params: Params) {
		let (device, queue, surface, surface_size) = params;
		// wgpu::RenderPass::draw(&mut self, vertices, instances);
		// wgpu::RenderPass::draw_indirect(&mut self, indirect_buffer, indirect_offset);

		let swap_chain = surface.get_current_texture().unwrap();

		// vk::Image & vk::ImageView = wgpu::Texture & wgpu::TextureView

		// vk::Format
		let surface_depth_format = TextureFormat::Depth32FloatStencil8;

		// vk::ImageCreateInfo
		let texture_desc = wgpu::TextureDescriptor {
			label: None,
			/// vk::ImageType
			dimension: TextureDimension::D2,
			format: surface_depth_format,
			/// vk::Extent3D
			size: Extent3d {
				width: surface_size.width,
				height: surface_size.height,
				depth_or_array_layers: 1,
			},
			mip_level_count: 1,
			sample_count: 1,
			/// vk::ImageUsageFlagBits
			/// RENDER_ATTACHMENT = eDepthStencilAttachment | eColorAttachment
			usage: TextureUsages::COPY_SRC.union(TextureUsages::RENDER_ATTACHMENT),
			// no arrayLayers, vk::ImageCreateFlags, vk::ImageTiling, vk::SharingMode, vk::ImageLayout or
		};
		// vk::Image depth_texture = device.createImage(vk::ImageCreateInfo)
		// vk::DeviceMemory depth_memory = device.allocateMemory(vk::MemoryAllocateInfo(..));
		// device.bindImageMemory(depth_texture, depth_memory, 0);
		let depth_texture = device.create_texture(&texture_desc);

		// vk::ImageViewCreateInfo
		let depth_texture_view_desc = wgpu::TextureViewDescriptor {
			label: None,
			/// vk::ImageViewType
			dimension: Some(TextureViewDimension::D2),
			/// vk::Format
			format: Some(surface_depth_format),

			/// vk::ImageSubresourceRange Start
			/// vk::ImageAspectFlagBits
			aspect: TextureAspect::All,
			base_mip_level: 0,
			mip_level_count: NonZeroU32::new(1),
			base_array_layer: 0,
			array_layer_count: NonZeroU32::new(1),
			// vk::ImageSubresourceRange End
			// No vk::ImageViewCreateFlags, vk::ComponentMapping
		};
		// vk::ImageView = device.createImageView(vk::ImageViewCreateInfos);
		let depth_texture_view = depth_texture.create_view(&depth_texture_view_desc);

		// vk::Buffer & vk::BufferView = wgpu::Buffer & wgpu::BufferView

		// vk::Pipeline = wgpu::RenderPipeline

		// vk::CommandBuffer
		let command_encoder =
			device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

		// command_encoder.copy_texture_to_texture(source, destination, copy_size);

		{
			// vk::RenderPassCreateInfo, vk::VkFramebufferCreateInfo
			let default_render_pass_descriptor = wgpu::RenderPassDescriptor {
				label: Some("Default Renderpass"),
				/// vk::AttachmentDescription
				color_attachments: &[Some(wgpu::RenderPassColorAttachment {
					ops: wgpu::Operations::default(),
					view: todo!(),
					resolve_target: todo!(),
				})],
				depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
					depth_ops: None,
					stencil_ops: Some(wgpu::Operations {
						store: false,
						..Default::default()
					}),
					view: todo!(),
				}),
			};
			// vk::CommandBuffer::beginRenderPass()
			let render_pass = command_encoder.begin_render_pass(&default_render_pass_descriptor);

			// ID3D12GraphicsCommandList::OMSetStencilRef
			// uint32_t vk::VkStencilOpState.reference (seperate for front and back)
			render_pass.set_stencil_reference(0);

			// vk::CommandBuffer::bindPipeline
			// render_pass.set_pipeline(pipeline)

			// vk::CommandBuffer::bindVertexBuffers
			// render_pass.set_vertex_buffer(slot, buffer_slice);

			// vk::CommandBuffer::bindIndexBuffer
			// render_pass.set_index_buffer(buffer_slice, IndexFormat::Uint16);

			// vk::CommandBuffer::setViewport
			// render_pass.set_viewport(x, y, w, h, min_depth, max_depth);

			// vk::CommandBuffer::setScissor
			// render_pass.set_scissor_rect(x, y, width, height)

			// blitImage

			// vk::CommandBuffer::drawIndexed
			// render_pass.draw_indexed(indices, base_vertex, instances)

			// VkSubpassDescription
			let default_render_bundle_encoder =
				device.create_render_bundle_encoder(&wgpu::RenderBundleEncoderDescriptor {
					label: Some("Default Render Bundle"),
					color_formats: &[Some(wgpu::TextureFormat::Rgba8UnormSrgb)],
					depth_stencil: Some(RenderBundleDepthStencil {
						format: TextureFormat::Bc3RgbaUnorm,
						depth_read_only: true,
						stencil_read_only: false,
					}),
					..Default::default()
				});

			let render_bundle =
				default_render_bundle_encoder.finish(&wgpu::RenderBundleDescriptor { label: None });

			// ID3D12GraphicsCommandList::ExecuteBundle(Iter<ID3D12GraphicsCommandList>)
			render_pass.execute_bundles([&render_bundle]);

			// vk::CommandBuffer::endRenderPass
			// drop(render_pass)
		}

		command_encoder.finish();

		// vk::SubmitInfo
		let submit_info = [command_encoder.finish()];

		queue.submit(submit_info);

		// vk::Fence = queue.on_submitted_work_done(callback)

		// queue.presentKHR(vk::PresentInfoKHR(..));
		swap_chain.present();

		// Missing: vkCmdPipelineBarrier, vk::Semaphore
	}

	fn buffers(params: Params) {
		let (device, queue, surface, surface_size) = params;

		let desc = BufferDescriptor {
			label: None,
			size: todo!(),
			usage: BufferUsages::VERTEX,
			mapped_at_creation: todo!(),
		};
		device.create_buffer(&desc);

		let desc = BufferInitDescriptor {
			label: None,
			contents: todo!(),
			usage: todo!(),
		};
		device.create_buffer_init(&desc);
	}

	///
	///
	/// BIND GROUP
	///
	///
	fn bind_group(params: Params) {
		let (device, queue, surface, surface_size) = params;

		// vk::PipelineLayout & vk::DescriptorSet = GPUPipelineLayout

		// vk::DescriptorSetLayoutCreateInfo
		let desc = BindGroupLayoutDescriptor {
			label: None,
			/// std::vector<vk::DescriptorSetLayoutBinding>
			entries: &[
				// vk::DescriptorSetLayoutBinding
				BindGroupLayoutEntry {
					binding: 0,
					/// vk::DescriptorType
					ty: BindingType::Buffer {
						ty: BufferBindingType::Uniform,
						has_dynamic_offset: todo!(),
						min_binding_size: todo!(),
					},
					/// vk::ShaderStageFlags (eVertex)
					visibility: ShaderStages::VERTEX,
					count: None,
				},
			],
		};
		let layout = device.create_bind_group_layout(&desc);

		let desc = BindGroupDescriptor {
			label: None,
			layout: &layout,
			entries: &[BindGroupEntry {
				binding: 0,
				resource: BindingResource::Buffer(todo!()),
			}],
		};
		let bind_group = device.create_bind_group(&desc);

		// vk::PipelineLayoutCreateInfo
		let pipeline_layout_desc = PipelineLayoutDescriptor {
			label: None,
			/// vk::DescriptorSetLayout
			bind_group_layouts: &[&layout],
			/// vk::PushConstantRange
			push_constant_ranges: &[
				// vk::PushConstantRange
				PushConstantRange {
					// vk::ShaderStageFlags stageFlags
					stages: ShaderStages::VERTEX,
					// offset + size
					range: (0..80),
				},
			],
			// Missing: vk::PipelineLayoutCreateFlags
		};
		// vk::PipelineLayout
		let pipeline_layout = device.create_pipeline_layout(&pipeline_layout_desc);
	}

	fn depth_buffer(params: Params) {
		let (device, queue, surface, surface_size) = params;
		device.push_error_scope(ErrorFilter::Validation);
		let format = TextureFormat::Depth32Float;
		let size = Extent3d {
			width: surface_size.width,
			height: surface_size.height,
			depth_or_array_layers: 1,
		};
		let texture_desc = TextureDescriptor {
			label: Some("Depth-buffer Texture"),
			size,
			mip_level_count: 1,
			sample_count: 1,
			dimension: TextureDimension::D2,
			format,
			usage: TextureUsages::RENDER_ATTACHMENT,
		};
		let depth_image = device.create_texture(&texture_desc);
		let view_desc = TextureViewDescriptor {
			aspect: TextureAspect::DepthOnly,
			..Default::default()
		};
		let depth_image_view = depth_image.create_view(&view_desc);
		let sampler = SamplerDescriptor {
			label: todo!(),
			address_mode_u: todo!(),
			address_mode_v: todo!(),
			address_mode_w: todo!(),
			mag_filter: todo!(),
			min_filter: todo!(),
			mipmap_filter: todo!(),
			lod_min_clamp: todo!(),
			lod_max_clamp: todo!(),
			compare: todo!(),
			anisotropy_clamp: todo!(),
			border_color: todo!(),
		};

		let depth_state = DepthStencilState {
			format,
			depth_write_enabled: true,
			depth_compare: CompareFunction::LessEqual,
			stencil: StencilState {
				front: StencilFaceState::VULKAN_DEFAULT,
				back: StencilFaceState::VULKAN_DEFAULT,
				..Default::default()
			},
			bias: DepthBiasState::default(),
		};

		let attachement = RenderPassDepthStencilAttachment {
			view: &depth_image_view,
			depth_ops: Some(Operations {
				load: LoadOp::Clear(1.0),
				store: true,
			}),
			stencil_ops: None,
		};

		let error = futures::executor::block_on(device.pop_error_scope());

		queue.write_texture(
			depth_image.as_image_copy(),
			todo!(),
			ImageDataLayout {
				offset: todo!(),
				bytes_per_row: todo!(),
				rows_per_image: todo!(),
			},
			size,
		)
	}

	///
	///
	/// PIPELINE
	///
	///
	fn pipeline(params: Params) {
		let (device, queue, surface, surface_size) = params;

		let module = device.create_shader_module(ShaderModuleDescriptor {
			label: todo!(),
			source: todo!(),
		});

		let spirv_module = unsafe {
			device.create_shader_module_spirv(&ShaderModuleDescriptorSpirV {
				label: todo!(),
				source: todo!(),
			})
		};

		// D3D12_GRAPHICS_PIPELINE_STATE_DESC
		// vk::GraphicsPipelineCreateInfo
		let render_pipeline_desc = RenderPipelineDescriptor {
			label: None,
			/// ID3D12RootSignature
			/// vk::PipelineLayout
			layout: None,
			/// vk::PipelineVertexInputStateCreateInfo
			vertex: VertexState {
				module: &module,
				entry_point: "main",
				buffers: &[
					// vk::VertexInputBindingDescription & [vk::VertexInputAttributeDescription]
					VertexBufferLayout {
						// uint32_t vk::VertexInputBindingDescription.stride
						array_stride: todo!(),
						// vk::VertexInputRate vk::VertexInputBindingDescription.inputRate (::eInstance)
						step_mode: VertexStepMode::Instance,
						attributes: &[
							// vk::VertexInputAttributeDescription
							VertexAttribute {
								// uint32_t offset
								offset: 0,
								// vk:Format format
								format: VertexFormat::Float32x3,
								// uint32_t location
								shader_location: 0,
							},
						],
					},
				],
				// Missing: https://www.w3.org/TR/webgpu/#typedefdef-gpupipelineconstantvalue
			},
			// vk::PipelineRasterizationStateCreateInfo
			fragment: Some(FragmentState {
				module: &module,
				entry_point: "main",
				targets: &[Some(ColorTargetState {
					format: todo!(),
					// vk::PipelineColorBlendAttachmentState
					blend: Some(BlendState {
						color: BlendComponent {
							// vk::BlendFactor srcColorBlendFactor_ (::eZero)
							src_factor: BlendFactor::Zero,
							// vk::BlendFactor dstColorBlendFactor_ (::eOne)
							dst_factor: BlendFactor::One,
							// vk::BlendOp colorBlendOp_ (::eAdd)
							operation: BlendOperation::Add,
						},
						alpha: BlendComponent {
							// vk::BlendFactor srcAlphaBlendFactor_ (::eZero)
							src_factor: BlendFactor::Zero,
							// vk::BlendFactor dstAlphaBlendFactor_ (::eZero)
							dst_factor: BlendFactor::Zero,
							// vk::BlendOp alphaBlendOp_ (::eAdd)
							operation: BlendOperation::Add,
						},
					}),
					// vk::ColorComponentFlags colorWriteMask_ (::eR|eG|eB|eA)
					write_mask: ColorWrites::ALL,
				})],
			}),
			/// vk::PipelineInputAssemblyStateCreateInfo &  
			/// Pick<vk::PipelineRasterizationStateCreateInfo,
			/// | polygonMode_
			/// | cullMode_
			/// | frontFace_>
			primitive: PrimitiveState {
				// vk::PrimitiveTopology (::eTriangleList)
				topology: PrimitiveTopology::TriangleList,
				// vk::PolygonMode (::eFill)
				polygon_mode: PolygonMode::Fill,
				// vk::CullModeFlagBits (::eNone)
				cull_mode: None,
				// vk::FrontFace (::eCounterClockwise)
				front_face: FrontFace::Ccw,
				// used with PrimitiveTopology::LineStrip, PrimitiveTopology::TriangleStrip
				// D3D12_INDEX_BUFFER_STRIP_CUT_VALUE
				strip_index_format: Some(IndexFormat::Uint16),
				// D3D12_CONSERVATIVE_RASTERIZATION_MODE
				// conservative:
				// unclipped_depth:

				// primitiveRestartEnable
				..Default::default()
			},
			// https://github.com/gpuweb/gpuweb/issues/118
			// D3D12_DEPTH_STENCIL_DESC
			// vk::PipelineDepthStencilStateCreateInfo
			// depthTestEnable = (depthCompare != Always)
			depth_stencil: Some(DepthStencilState {
				format: todo!(),
				// BOOL DepthEnable
				// vk::Bool32 depthWriteEnable_
				depth_write_enabled: true,
				// D3D12_COMPARISON_FUNC DepthFunc
				/// vk::CompareOp depthCompareOp_ (::eLessOrEqual)
				depth_compare: CompareFunction::LessEqual,
				stencil: StencilState {
					// D3D12_DEPTH_STENCILOP_DESC FrontFace
					// vk::StencilOpState front_
					front: StencilFaceState::VULKAN_DEFAULT,
					// D3D12_DEPTH_STENCILOP_DESC BackFace
					// vk::StencilOpState back_
					back: StencilFaceState::VULKAN_DEFAULT,
					// UINT8 StencilReadMask
					read_mask: 0,
					// UINT8 StencilWriteMask
					write_mask: 0,
				},
				/// Pick<vk::PipelineRasterizationStateCreateInfo,
				/// depthBiasConstantFactor_
				/// |depthBiasSlopeFactor_
				/// | depthBiasClamp>
				bias: DepthBiasState {
					// vk::PipelineRasterizationStateCreateInfo::depthBiasConstantFactor_
					constant: 0,
					// vk::PipelineRasterizationStateCreateInfo::depthBiasSlopeFactor_
					slope_scale: 0.0,
					// vk::PipelineRasterizationStateCreateInfo::depthBiasClamp
					clamp: 0.0,
				},
			}),
			// vk::PipelineMultisampleStateCreateInfo
			multisample: MultisampleState {
				// D3D12_BLEND_DESC.AlphaToCoverageEnable
				// vk::Bool32 alphaToCoverageEnable
				alpha_to_coverage_enabled: false,
				// UINT DXGI_SAMPLE_DESC.Count
				// vk::SampleCountFlagBits rasterizationSamples
				count: 1,
				// UINT D3D12_GRAPHICS_PIPELINE_STATE_DESC.SampleMask
				// vk::SampleMask pSampleMask (Vulkan allows multiple sampleMasks)
				mask: todo!(),
			},
			// Only needed for VR or Cubemaps
			multiview: None,
			// # Missing Stages:
			// - PipelineTessellationStateCreateInfo
			// - PipelineViewportStateCreateInfo
			// - PipelineColorBlendStateCreateInfo
			// - PipelineDynamicStateCreateInfo
		};
		// vk::Pipeline = (vk::Device::createGraphicsPipeline).value
		let pipeline = device.create_render_pipeline(&render_pipeline_desc);
	}
}

mod engine {
	use std::{
		error::Error,
		fmt::Debug,
		path::{Path, PathBuf},
	};

	use wgpu::{util::DeviceExt, PipelineLayout};
	use winit::window::Window;

	use crate::learn;

	#[derive(Debug, Copy, Clone)]
	enum SampleCount {
		Bit1 = 0x00000001,
		Bit2 = 0x00000002,
		Bit4 = 0x00000004,
		Bit8 = 0x00000008,
		Bit16 = 0x00000010,
		Bit32 = 0x00000020,
		Bit64 = 0x00000040,
	}

	utils::with_offsets! {
		#[repr(C)]
		#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
		pub struct Vertex {
			position: glam::Vec3,
			normal: glam::Vec3,
			color: glam::Vec3,
		}
	}

	impl Vertex {
		fn description<'a>() -> wgpu::VertexBufferLayout<'a> {
			use std::mem;

			wgpu::VertexBufferLayout {
				array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
				step_mode: wgpu::VertexStepMode::Vertex,
				attributes: &[
					wgpu::VertexAttribute {
						offset: Vertex::POSITION_OFFSET as wgpu::BufferAddress,
						shader_location: 0,
						format: wgpu::VertexFormat::Float32x3,
					},
					wgpu::VertexAttribute {
						offset: Vertex::NORMAL_OFFSET as wgpu::BufferAddress,
						// offset: mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
						// offset: memoffset::offset_of!(Vertex, normal) as wgpu::BufferAddress,
						shader_location: 1,
						format: wgpu::VertexFormat::Float32x3,
					},
					wgpu::VertexAttribute {
						offset: Vertex::COLOR_OFFSET as wgpu::BufferAddress,
						// offset: mem::size_of::<[f32; 6]>() as wgpu::BufferAddress,
						// offset: memoffset::offset_of!(Vertex, color) as wgpu::BufferAddress,
						shader_location: 2,
						format: wgpu::VertexFormat::Float32x3,
					},
				],
			}
		}
	}

	#[derive(Debug)]
	struct Mesh {
		label: Option<String>,
		vertices: Vec<Vertex>,
		buffer: wgpu::Buffer,
	}

	struct PipelineBuilder<'a> {
		device: &'a wgpu::Device,
		targets: Vec<Option<wgpu::ColorTargetState>>,
		vertex_module: &'a wgpu::ShaderModule,
		fragment_module: &'a wgpu::ShaderModule,
		polygon_mode: wgpu::PolygonMode,
		topology: wgpu::PrimitiveTopology,
		layout: Option<&'a wgpu::PipelineLayout>,
		buffers: Vec<wgpu::VertexBufferLayout<'a>>,
	}

	impl<'a> PipelineBuilder<'a> {
		fn create_vertex_state(&self) -> wgpu::VertexState {
			wgpu::VertexState {
				module: self.vertex_module,
				entry_point: "main",
				buffers: &self.buffers[..],
			}
		}

		fn create_fragemnt_state(&self) -> wgpu::FragmentState {
			wgpu::FragmentState {
				module: self.fragment_module,
				entry_point: "main",
				targets: &self.targets,
			}
		}

		fn create_primitive_state(&self) -> wgpu::PrimitiveState {
			wgpu::PrimitiveState {
				topology: self.topology,
				front_face: wgpu::FrontFace::Cw,
				cull_mode: None,
				polygon_mode: self.polygon_mode,
				..Default::default()
			}
		}

		fn build(self) -> wgpu::RenderPipeline {
			self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
				label: None,
				layout: self.layout,
				vertex: self.create_vertex_state(),
				fragment: Some(self.create_fragemnt_state()),
				primitive: self.create_primitive_state(),
				depth_stencil: None,
				multisample: wgpu::MultisampleState::default(),
				multiview: None,
			})
		}
	}

	struct Context {
		instance: wgpu::Instance,
		adapter: wgpu::Adapter,
		device: wgpu::Device,
		surface: wgpu::Surface,
		queue: wgpu::Queue,
	}

	impl Context {
		fn new(window: &Window) -> Self {
			let (instance, surface, adapter, device, queue) =
				futures::executor::block_on(Self::init_wgpu(&window));

			Self {
				instance,
				surface,
				adapter,
				device,
				queue,
			}
		}

		async fn init_wgpu(window: &Window) -> WebGPUResources {
			use wgpu::*;

			let instance = Instance::new(Backends::all());
			let adapters = instance
				.enumerate_adapters(Backends::all())
				.inspect(|adapter| {
					dbg!(adapter);
				})
				.collect::<Vec<_>>();

			let default_adapter =
				instance.request_adapter(&RequestAdapterOptions::default()).await.unwrap();
			let (device, queue) = default_adapter
				.request_device(&DeviceDescriptor::default(), None)
				.await
				.unwrap();
			let surface = unsafe { instance.create_surface(&window) };

			let formats = surface.get_supported_formats(&default_adapter);
			let alpha_modes = surface.get_supported_alpha_modes(&default_adapter);

			let formats = dbg!(formats);

			let surface_config = SurfaceConfiguration {
				usage: TextureUsages::RENDER_ATTACHMENT,
				format: formats[0],
				width: window.inner_size().width,
				height: window.inner_size().height,
				present_mode: PresentMode::AutoVsync,
				alpha_mode: alpha_modes[0],
			};
			surface.configure(&device, &surface_config);

			(instance, surface, default_adapter, device, queue)
		}

		fn init_pipelines(&self) -> (wgpu::RenderPipeline, wgpu::PipelineLayout) {
			let triangle_vertex = load_shader_module(
				&self.device,
				format!(
					"{}{}",
					env!("CARGO_MANIFEST_DIR"),
					"/../assets/shaders/triangle/shader.vert.modern.wgsl"
				),
				Some("Triangle Vertex"),
			)
			.expect("Error when building the triangle vertex shader module");

			let triangle_fragment = load_shader_module(
				&self.device,
				format!(
					"{}{}",
					env!("CARGO_MANIFEST_DIR"),
					"/../assets/shaders/triangle/shader.frag.modern.wgsl"
				),
				Some("Triangle Fragment"),
			)
			.expect("Error when building the triangle fragment shader module");

			let triangle_pipeline_layout =
				self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
					label: None,
					bind_group_layouts: &[],
					push_constant_ranges: &[],
				});

			let targets = vec![Some(wgpu::ColorTargetState {
				format: self.surface.get_supported_formats(&self.adapter)[0],
				blend: Some(wgpu::BlendState::REPLACE),
				write_mask: wgpu::ColorWrites::ALL,
			})];

			let triangle_pipeline = PipelineBuilder {
				device: &self.device,
				targets,
				vertex_module: &triangle_vertex,
				fragment_module: &triangle_fragment,
				polygon_mode: wgpu::PolygonMode::Fill,
				topology: wgpu::PrimitiveTopology::TriangleList,
				layout: Some(&triangle_pipeline_layout),
				buffers: vec![],
			}
			.build();

			(triangle_pipeline, triangle_pipeline_layout)
		}

		fn init_mesh_pipeline(&self) -> (wgpu::RenderPipeline, wgpu::PipelineLayout) {
			let vertex = load_shader_module(
				&self.device,
				format!(
					"{}{}",
					env!("CARGO_MANIFEST_DIR"),
					"/../assets/shaders/triangle/shader.vert.modern.wgsl"
				),
				Some("Triangle Vertex"),
			)
			.expect("Error when building the triangle vertex shader module");

			let fragment = load_shader_module(
				&self.device,
				format!(
					"{}{}",
					env!("CARGO_MANIFEST_DIR"),
					"/../assets/shaders/triangle/shader.frag.modern.wgsl"
				),
				Some("Triangle Fragment"),
			)
			.expect("Error when building the triangle fragment shader module");

			let pipeline_layout =
				self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
					label: None,
					bind_group_layouts: &[],
					push_constant_ranges: &[],
				});

			let targets = vec![Some(wgpu::ColorTargetState {
				format: self.surface.get_supported_formats(&self.adapter)[0],
				blend: Some(wgpu::BlendState::REPLACE),
				write_mask: wgpu::ColorWrites::ALL,
			})];

			let pipeline = PipelineBuilder {
				device: &self.device,
				targets,
				vertex_module: &vertex,
				fragment_module: &fragment,
				polygon_mode: wgpu::PolygonMode::Fill,
				topology: wgpu::PrimitiveTopology::TriangleList,
				layout: Some(&pipeline_layout),
				buffers: vec![Vertex::description()],
			}
			.build();

			(pipeline, pipeline_layout)
		}

		fn load_meshes(&self) -> Mesh {
			let label = "Green Triangle".to_string();

			let vertices = vec![
				Vertex {
					position: glam::vec3(1.0, 1.0, 0.0),
					normal: glam::Vec3::default(),
					color: glam::vec3(0.0, 1.0, 0.0),
				},
				Vertex {
					position: glam::vec3(-1.0, 1.0, 0.0),
					normal: glam::Vec3::default(),
					color: glam::vec3(0.0, 1.0, 0.0),
				},
				Vertex {
					position: glam::vec3(0.0, -1.0, 0.0),
					normal: glam::Vec3::default(),
					color: glam::vec3(0.0, 1.0, 0.0),
				},
			];

			let buffer = self.upload_mesh(Some(&label), &vertices);

			Mesh {
				label: Some(label),
				vertices,
				buffer,
			}
		}

		fn upload_mesh(&self, label: Option<&str>, mesh: &[Vertex]) -> wgpu::Buffer {
			self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
				label,
				contents: &bytemuck::cast_slice(mesh),
				usage: wgpu::BufferUsages::VERTEX,
			})
		}
	}

	pub struct Engine {
		window: winit::window::Window,
		event_loop: Option<winit::event_loop::EventLoop<()>>,
		ctx: Context,
		// triangle_pipeline: wgpu::RenderPipeline,
		mesh_pipeline: wgpu::RenderPipeline,
		triangle_mesh: Mesh,
		frame_number: u32,
	}

	type WebGPUResources = (
		wgpu::Instance,
		wgpu::Surface,
		wgpu::Adapter,
		wgpu::Device,
		wgpu::Queue,
	);

	pub type EngineResult<T> = Result<T, Box<dyn Error>>;

	impl Engine {
		pub fn init() -> Self {
			// init_vulkan();
			// init_swapchain();
			let event_loop = winit::event_loop::EventLoop::new();
			let window = winit::window::WindowBuilder::new()
				.with_title("WebGPU Engine")
				.with_inner_size(winit::dpi::PhysicalSize::new(1700, 900))
				.build(&event_loop)
				.unwrap();

			let ctx = Context::new(&window);

			// init_commands();

			// init_default_renderpass();
			// Moved to draw because WebGPU needs a new Renderpass on each draw

			// init_framebuffers();

			// init_sync_structures();
			// doesn't exist in WebGPU

			// init_pipelines();
			// let (triangle_pipeline, triangle_pipeline_layout) = ctx.init_pipelines();
			let (mesh_pipeline, mesh_pipeline_layout) = ctx.init_mesh_pipeline();

			let triangle_mesh = ctx.load_meshes();

			Self {
				window,
				event_loop: Some(event_loop),
				ctx,
				triangle_mesh,
				mesh_pipeline,
				// triangle_pipeline,
				frame_number: 0,
			}
		}

		pub fn run(mut self) {
			use winit::event::*;

			self.event_loop
				.take()
				.unwrap()
				.run(move |event, _window, control_flow| match event {
					Event::WindowEvent { event, .. } => match event {
						WindowEvent::CloseRequested => {
							*control_flow = winit::event_loop::ControlFlow::Exit;
						},
						_ => {},
					},
					Event::RedrawRequested(_window_id) => {
						self.draw();
					},
					_ => {},
				})
		}

		fn load_shader_module(
			&self,
			path: impl Into<PathBuf> + Debug,
			label: Option<&str>,
		) -> EngineResult<wgpu::ShaderModule> {
			load_shader_module(&self.ctx.device, path, label)
		}

		fn draw(&mut self) -> Result<(), Box<dyn Error>> {
			let swap_chain = self.ctx.surface.get_current_texture()?;
			let view = swap_chain.texture.create_view(&wgpu::TextureViewDescriptor::default());

			let mut main_command_encoder =
				self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
					label: Some("Main Command Encoder"),
				});
			{
				let mut render_pass = init_default_renderpass(&mut main_command_encoder, &view);
				render_pass.set_pipeline(&self.mesh_pipeline);
				render_pass.set_vertex_buffer(0, self.triangle_mesh.buffer.slice(..));
				render_pass.draw(0..self.triangle_mesh.vertices.len() as u32, 0..1);
			}

			self.ctx.queue.submit([main_command_encoder.finish()]);
			swap_chain.present();
			self.frame_number += 1;
			Ok(())
		}
	}

	fn init_default_renderpass<'a>(
		cmd_encoder: &'a mut wgpu::CommandEncoder,
		view: &'a wgpu::TextureView,
	) -> wgpu::RenderPass<'a> {
		cmd_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
			label: Some("Default Renderpass"),
			color_attachments: &[Some(wgpu::RenderPassColorAttachment {
				view,
				resolve_target: None,
				ops: wgpu::Operations {
					load: wgpu::LoadOp::Clear(wgpu::Color::default()),
					store: true,
				},
			})],
			depth_stencil_attachment: None,
		})
	}

	fn load_shader_module(
		device: &wgpu::Device,
		path: impl Into<PathBuf> + Debug,
		label: Option<&str>,
	) -> EngineResult<wgpu::ShaderModule> {
		use std::fs;

		let source = wgpu::ShaderSource::Wgsl(fs::read_to_string(path.into())?.into());
		let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label, source });

		Ok(module)
	}

	pub mod utils {
		macro_rules! with_offsets {
			(
				#[repr(C)]
				$(#[$struct_meta:meta])*
				$struct_vis:vis struct $StructName:ident {
					$(
						$(#[$field_meta:meta])*
						$field_vis:vis $field_name:ident: $field_ty:ty
					),*
					$(,)?
				}
			) => (
				#[repr(C)]
				$(#[$struct_meta])*
				$struct_vis struct $StructName {
					$(
						$(#[$field_meta])*
						$field_vis $field_name: $field_ty,
					)*
				}

				#[allow(nonstandard_style)]
				const _: () = {
					paste::paste! {
						pub struct StructOffsets {
							$(
								$field_vis [<$field_name:upper>]: usize,
							)*
						}
						struct Helper;
						impl $StructName {
							$(
								pub const [<$field_name:upper _OFFSET>]: usize = Helper::$field_name;
							)*

							pub const offset_to: StructOffsets = StructOffsets {
								$(
									[<$field_name:upper>]: Helper::$field_name,
								)*
							};
						}
					}
					const END_OF_PREV_FIELD: usize = 0;
					$crate::engine::utils::with_offsets! {
						@names [ $($field_name)* ]
						@tys [ $($field_ty ,)*]
					}
				};
			);
			(
				@names []
				@tys []
			) => ();
			(
				@names [$field_name:ident $($other_names:tt)*]
				@tys [$field_ty:ty , $($other_tys:tt)*]
			) => (
				impl Helper {
					const $field_name: usize = {
						let align = core::mem::align_of::<$field_ty>();
						let trail = END_OF_PREV_FIELD % align;

						0 + END_OF_PREV_FIELD + (align - trail) * [1, 0][(trail == 0) as usize]
					};
				}
				const _: () = {
					const END_OF_PREV_FIELD: usize = Helper::$field_name + core::mem::size_of::<$field_ty>();
					$crate::engine::utils::with_offsets! {
						@names [$($other_names)*]
						@tys [$($other_tys)*]
					}
				};
			);
		}

		pub(crate) use with_offsets;
	}

	#[cfg(test)]
	mod test {

		#[test]
		fn test() {}
	}
}
