#version 400
#define INVPOL_DEGREE 18

// Input vertex data, different for all executions of this shader.
layout(location = 0) in vec3 vertexPos;
layout(location = 1) in vec3 vertexColor;
layout(location = 2) in int vertexId;

// Output data ; will be interpolated for each fragment.
out VS_OUT {
    vec3 color;
	int inst_id;
	float depth;
} vs_out;

// Values that stay constant for the whole mesh.
uniform mat4 MV;
uniform double ocam_invpol[INVPOL_DEGREE];
uniform dvec3 ocam_affine;
uniform dvec2 ocam_center_off;
uniform float ocam_theta_thresh;
uniform float far;
//uniform vec3 OFFSET;
void main(){

	vec4 vertexPosMV = MV * vec4(vertexPos, 1);
	float xynorm = length(vertexPosMV.xy);
	double theta = -atan(vertexPosMV.z, xynorm);
	double cur_theta = theta;
	double rho = ocam_invpol[0];
	for (int i=1; i<INVPOL_DEGREE; ++i)
	{
		rho += cur_theta*ocam_invpol[i];
		cur_theta *= theta;
	}

	dvec2 uv = vertexPosMV.xy*rho/xynorm;
	dvec2 res = dvec2(uv.x + uv.y*ocam_affine.z, dot(uv, ocam_affine.yx)) + ocam_center_off.yx;

	gl_Position = vec4(res,
	                   length(vertexPosMV)/far*2-1,
					   1.0);

	vs_out.color = vertexColor;
	vs_out.inst_id = vertexId;
	vs_out.depth = abs(vertexPosMV.z);
}