#version 330 core

in vec3 v_world_pos;

out vec4 fragColor;

uniform vec3 u_fill;
uniform vec3 u_border;
uniform float u_alpha;
uniform float u_is_border;

void main() {
    // The capture sensor / image plane is intentionally NOT clipped by the
    // viewport clip planes — those only carve through rays and lens elements.
    vec3 col = mix(u_fill, u_border, u_is_border);
    float a = mix(u_alpha, 1.0, u_is_border);
    fragColor = vec4(col, a);
}
