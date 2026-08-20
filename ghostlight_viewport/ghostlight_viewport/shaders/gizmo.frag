#version 330 core

in vec3 v_normal;
in vec3 v_color;

out vec4 fragColor;

uniform float u_highlight;

void main() {
    vec3 L = normalize(vec3(0.4, 0.6, 0.7));
    float lambert = clamp(dot(normalize(v_normal), L), 0.2, 1.0);
    vec3 col = v_color * lambert;
    col = mix(col, vec3(1.0, 0.6, 0.1), u_highlight);
    fragColor = vec4(col, 1.0);
}
