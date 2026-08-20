#version 330 core

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;     // bound by widget VAO setup; unused since
                                           // the flat-shaded lens.frag dropped Gooch+Fresnel
layout(location = 2) in float a_kind;

uniform mat4 u_view_proj;
uniform mat4 u_model;

out vec3 v_world_pos;
out float v_kind;

void main() {
    vec4 world = u_model * vec4(a_position, 1.0);
    v_world_pos = world.xyz;
    v_kind = a_kind;
    gl_Position = u_view_proj * world;
}
