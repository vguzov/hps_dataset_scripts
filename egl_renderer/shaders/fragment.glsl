#version 330 core

// Interpolated values from the vertex shaders
in vec3 vcolor;
flat in int frag_inst_id;

// Ouput data
layout(location = 0) out vec3 color;
layout(location = 1) out int pix_inst_id;



// Values that stay constant for the whole mesh.
//uniform sampler2D myTextureSampler;

void main(){

	// Output color = color of the texture at the specified UV
	color = vcolor;
	pix_inst_id = frag_inst_id;
}