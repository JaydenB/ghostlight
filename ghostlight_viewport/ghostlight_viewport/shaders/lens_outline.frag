#version 330 core

in vec3 v_world_pos;
in float v_side_px;            // signed perpendicular distance from line centre, in pixels

out vec4 fragColor;

uniform vec3 u_color;
uniform float u_alpha;
uniform vec4 u_clip_plane;
uniform vec4 u_clip_plane_b;
uniform float u_line_half_width_px;
// Small bias so cross-section segments — which lie exactly on the slicing
// plane analytically — survive the same plane's discard test in spite of
// float-precision drift through the transform pipeline.  Tuned to scene
// scale (mm); raise if cross-section strokes start dropping out at extreme
// zoom-out / very large prescriptions.
const float CLIP_EPS = 1e-3;

void main() {
    if (dot(u_clip_plane.xyz, u_clip_plane.xyz) > 0.5) {
        if (dot(v_world_pos, u_clip_plane.xyz) + u_clip_plane.w > CLIP_EPS) {
            discard;
        }
    }
    if (dot(u_clip_plane_b.xyz, u_clip_plane_b.xyz) > 0.5) {
        if (dot(v_world_pos, u_clip_plane_b.xyz) + u_clip_plane_b.w > CLIP_EPS) {
            discard;
        }
    }

    // Analytic-coverage anti-aliasing: full opacity within the nominal
    // half-width, smoothstep falloff across a 1-pixel feather band at
    // each edge.  Without this the quad reads as a hard-edged ribbon and
    // shows the staircase the user spotted.
    float dist = abs(v_side_px);
    float aa = 1.0 - smoothstep(
        u_line_half_width_px - 0.5,
        u_line_half_width_px + 0.5,
        dist
    );
    if (aa <= 0.0) {
        discard;
    }
    fragColor = vec4(u_color, u_alpha * aa);
}
