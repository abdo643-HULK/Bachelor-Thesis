// #![feature(dropck_eyepatch)]
// #![feature(specialization)]

pub mod app;
// pub mod assets;
// pub mod camera;
// pub mod controls;
// mod gltf;
mod loader;
// mod mesh;
// pub mod model;
// pub mod renderer;
// pub mod triangle;
// pub mod triangle_simple;
// pub mod vertex;

use app::{App, AppOptions};
use loader::GLTFLoader;

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

    let loader = GLTFLoader::new(path, loader::AnimationLoop::Disable);

    // let mut controls = controls::orbital::OrbitControls::default();
    // controls.object.update();

    App::init(AppOptions {
        title: Some(title),
        ..Default::default()
    })
    .add_system(Box::new(loader))
    .run();

    // triangle::run();
    // triangle_simple::run();

    return Ok(());
}

// pub struct Inspector<'a, T: std::fmt::Debug>(&'a T);

// unsafe impl<'a, #[may_dangle] T: std::fmt::Debug> Drop for Inspector<'a, T> {
//     fn drop(&mut self) {
//         println!("Inspector({:#?})", self.0);
//     }
// }

mod learn {
    use wgpu::{*, util::RenderEncoder};
    use winit::dpi::LogicalSize;

    const VULKAN_DEFAULT_STENCIL_OP: StencilFaceState = StencilFaceState {
        compare: CompareFunction::Never,
        fail_op: StencilOperation::Keep,
        depth_fail_op: StencilOperation::Keep,
        pass_op: StencilOperation::Keep,
    };

    type Params = (&Device, &Queue, &Surface, LogicalSize<u32>);
    // wgpu::Surface = vk::Surface & vk::SwapChain
    // wgpu::Device = vk::Device & vk::CommandPool. It allocates commandbuffers
    fn learn(params: Params) {
        let (device, queue, surface, surface_size) = params;
        // wgpu::RenderPass::draw(&mut self, vertices, instances);
        // wgpu::RenderPass::draw_indirect(&mut self, indirect_buffer, indirect_offset);

        let swap_chain = surface.get_current_texture().unwrap();

        // vk::Image & vk::ImageView = wgpu::Texture & wgpu::TextureView

        /// vk::Format
        let surface_depth_format = TextureFormat::Depth32FloatStencil8;

        /// vk::ImageCreateInfo
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
            /// no arrayLayers, vk::ImageCreateFlags, vk::ImageTiling, vk::SharingMode, vk::ImageLayout or
        };
        /// vk::Image depth_texture = device.createImage(vk::ImageCreateInfo)
        /// vk::DeviceMemory depth_memory = device.allocateMemory(vk::MemoryAllocateInfo(..));
        /// device.bindImageMemory(depth_texture, depth_memory, 0);
        let depth_texture = device.create_texture(&texture_desc);

        /// vk::ImageViewCreateInfo
        let depth_texture_view_desc = wgpu::TextureViewDescriptor {
            label: None,
            /// vk::ImageViewType
            dimension: Some(TextureViewDimension::D2),
            /// vk::Format
            format: Some(surface_depth_format),

            /// vk::ImageSubresourceRange Start
            /// vk::ImageAspectFlagBits
            aspect: TextureAspect::DepthOnly | TextureAspect::StencilOnly,
            base_mip_level: 0,
            mip_level_count: 1,
            base_array_layer: 0,
            array_layer_count: 1,
            /// vk::ImageSubresourceRange End
            /// No vk::ImageViewCreateFlags, vk::ComponentMapping
        };
        /// vk::ImageView = device.createImageView(vk::ImageViewCreateInfos);
        let depth_texture_view = depth_texture.create_view(&depth_texture_view_desc);

        /// vk::Buffer & vk::BufferView = wgpu::Buffer & wgpu::BufferView

        /// vk::Pipeline = wgpu::RenderPipeline

        /// vk::CommandBuffer
        let command_encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

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

        // command_encoder.copy_texture_to_texture(source, destination, copy_size);

        {
            /// vk::RenderPassCreateInfo, vk::VkFramebufferCreateInfo
            let default_render_pass_descriptor = wgpu::RenderPassDescriptor {
                label: Some("Default Renderpass"),
                /// VkAttachmentDescription
                color_attachments: [Some(&wgpu::RenderPassColorAttachment {
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
            let render_pass = command_encoder.begin_render_pass(&default_render_pass_descriptor);

            // VkSubpassDescription
            let render_bundle =
                default_render_bundle_encoder.finish(&wgpu::RenderBundleDescriptor { label });

            render_pass.execute_bundles([render_bundle]);

            // vk::CommandBuffer::bindPipeline
            // render_pass.set_pipeline(pipeline)
            // vk::CommandBuffer::bindVertexBuffers
            // render_pass.set_vertex_buffer(slot, buffer_slice);
            // vk::CommandBuffer::bindIndexBuffer
            // render_pass.set_index_buffer(buffer_slice, index_format);
            // vk::CommandBuffer::setViewport
            // render_pass.set_viewport(x, y, w, h, min_depth, max_depth);
            // vk::CommandBuffer::setScissor
            // render_pass.set_scissor_rect(x, y, width, height)
            // blitImage

            // render_pass.draw_indexed(indices, base_vertex, instances)   
            // vk::CommandBuffer::endRenderPass
            // drop(render_pass)          
        }

        command_encoder.finish();

        /// vk::SubmitInfo
        let submit_info = [command_encoder];

        // vk::Fence = queue.on_submitted_work_done(callback)

        // queue.presentKHR(vk::PresentInfoKHR(..));
        swap_chain.present();

        /// Missing: vkCmdPipelineBarrier, vk::Semaphore 
    }

    fn bind_group(params: Params) {
        let (device, queue, surface, surface_size) = params;
        /// vk::PipelineLayout & vk::DescriptorSet = GPUPipelineLayout

        /// vk::DescriptorSetLayoutCreateInfo
        let desc = BindGroupLayoutDescriptor {
            label: None,
            /// std::vector<vk::DescriptorSetLayoutBinding>
            entries: &[
                /// vk::DescriptorSetLayoutBinding
                BindGroupLayoutEntry {
                    binding: 0,
                    /// vk::DescriptorType
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: (),
                        min_binding_size: (),
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

        /// vk::PipelineLayoutCreateInfo
        let pipeline_layout_desc = PipelineLayoutDescriptor {
            label: None,
            /// vk::DescriptorSetLayout
            bind_group_layouts: &[
                &layout,
            ],
            /// vk::PushConstantRange
            push_constant_ranges: &[],
            /// Missing: vk::PipelineLayoutCreateFlags
        };
        // vk::PipelineLayout
        let pipeline_layout = device.create_pipeline_layout(&pipeline_layout_desc);
    }

    fn pipeline(params: Params) {
        let (device, queue, surface, surface_size) = params;

        device.create_shader_module(ShaderModuleDescriptor {
            label: todo!(),
            source: todo!(),
        });

        // vk::GraphicsPipelineCreateInfo
        let render_pipeline_desc = RenderPipelineDescriptor {
            label: None,
            /// vk::PipelineLayout
            layout: todo!(),
            /// vk::PipelineVertexInputStateCreateInfo
            vertex: todo!(),
            /// vk::PipelineInputAssemblyStateCreateInfo &  vk::PipelineRasterizationStateCreateInfo
            primitive: PrimitiveState {
                // vk::PrimitiveTopology (::eTriangleList)
                topology: PrimitiveTopology::TriangleList,
                // vk::PolygonMode (::eFill)
                polygon_mode: PolygonMode::Fill,
                // vk::CullModeFlagBits (::eNone)
                cull_mode: None,
                // vk::FrontFace (::eCounterClockwise)
                front_face: FrontFace::Ccw,

                ..Default::default()
            },
            // vk::PipelineDepthStencilStateCreateInfo
            depth_stencil: Some(DepthStencilState { 
                format: todo!(), 
                // Bool32 depthWriteEnable_
                depth_write_enabled: true, 
                /// vk::CompareOp depthCompareOp_ (::eLessOrEqual)
                depth_compare: CompareFunction::LessEqual, 
                stencil: StencilState { 
                    // vk::StencilOpState front_
                    front: VULKAN_DEFAULT_STENCIL_OP,
                    // vk::StencilOpState back_ 
                    back: VULKAN_DEFAULT_STENCIL_OP, 
                    read_mask: 0,     
                    write_mask: 0 
                }, 
                bias: DepthBiasState { 
                    constant: todo!(), 
                    slope_scale: todo!(), 
                    clamp: todo!() 
                }
            }),
            // vk::PipelineMultisampleStateCreateInfo
            multisample: MultisampleState { 
                count: 1,
                ..Default::default()
            },
            // vk::PipelineRasterizationStateCreateInfo
            fragment: Some(FragmentState { 
                module: todo!(), 
                entry_point: todo!(),
                targets: &[
                    Some(ColorTargetState {
                        format: todo!(),
                        // vk::PipelineColorBlendAttachmentState
                        blend: Some(
                            BlendState {
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
                    })
                ] 
            }),
            multiview: todo!(),
            /// # Missing Stages: 
            /// - PipelineTessellationStateCreateInfo 
            /// - PipelineViewportStateCreateInfo
            /// - PipelineColorBlendStateCreateInfo
            /// - PipelineDynamicStateCreateInfo
        };
        // vk::Pipeline = (vk::Device::createGraphicsPipeline).value
        let pipeline = device.create_render_pipeline(&render_pipeline_desc);
    }
}
