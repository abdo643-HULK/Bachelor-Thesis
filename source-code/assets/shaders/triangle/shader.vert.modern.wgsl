struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>
};

struct Vertex {
    @location(0) pos: vec3<f32>,
	@location(1) normal: vec3<f32>,
	@location(2) color: vec3<f32>,
};

@vertex
fn main(
    @builtin(vertex_index) in_vertex_index: u32,
    in: Vertex,
) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(in.pos, 1.0);
    out.color = vec4<f32>(in.color, 1.0);
    return out;
}
