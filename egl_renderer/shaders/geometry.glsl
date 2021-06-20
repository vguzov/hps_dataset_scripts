#version 330 core
layout (points) in;
layout (triangle_strip, max_vertices = 4) out;
out vec3 vcolor;
flat out int frag_inst_id;

in VS_OUT {
    vec3 color;
    int inst_id;
    float depth;
} gs_in[];

void main() {
    vec4 position = gl_in[0].gl_Position;
    float size_mul = 1./(1+0.2*gs_in[0].depth)*position.w;
    vcolor = gs_in[0].color;
    frag_inst_id = gs_in[0].inst_id;
    gl_Position = position + vec4(-0.01, -0.01, 0.0, 0.0)*size_mul;
    EmitVertex();

    gl_Position = position + vec4(0.01, -0.01, 0.0, 0.0)*size_mul;
    EmitVertex();

    gl_Position = position + vec4(-0.01, 0.01, 0.0, 0.0)*size_mul;
    EmitVertex();

    gl_Position = position + vec4(0.01, 0.01, 0.0, 0.0)*size_mul;
    EmitVertex();

    EndPrimitive();
}
