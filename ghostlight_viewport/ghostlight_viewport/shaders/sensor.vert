#version 330 core

layout(location = 0) in vec3 a_position;

uniform mat4 u_view_proj;

out vec3 v_world_pos;

void main() {
    v_world_pos = a_position;
    gl_Position = u_view_proj * vec4(a_position, 1.0);
}
