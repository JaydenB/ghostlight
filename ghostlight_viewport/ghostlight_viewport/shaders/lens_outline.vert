#version 330 core

// Per-vertex layout for screen-space miter-joined quad-expanded lines.
// Each polyline vertex emits 2 GL output vertices (top + bottom of the
// stroke) and carries its two polyline neighbours so the shader can
// compute the bisector direction shared by the segments on either side.
// Disjoint segments degenerate by setting prev == this and/or next ==
// this; the shader detects that and falls back to a perpendicular butt
// cap so the same code path handles closed rim loops and standalone
// cross-section segments.
layout(location = 0) in vec3 a_this;      // this polyline vertex
layout(location = 1) in vec3 a_prev;      // polyline vertex before this one
layout(location = 2) in vec3 a_next;      // polyline vertex after this one
layout(location = 3) in float a_side;     // +1 / -1 — perpendicular offset sign

uniform mat4 u_view_proj;
uniform mat4 u_model;
uniform vec2 u_viewport_size_px;          // (width, height) of the destination FBO
uniform float u_line_half_width_px;       // half stroke thickness, in framebuffer pixels

out vec3 v_world_pos;
out float v_side_px;                      // signed perpendicular distance from line centre

// Extra pixels of geometry past the nominal stroke width so the fragment
// shader has room to smoothstep alpha to zero without the falloff being
// clipped by the quad boundary.
const float AA_EDGE_PX = 0.5;

// Clamp the miter extension factor so very sharp turns don't spike the
// stroke outward.  cos(angle/2) = 0.1 corresponds to ~169° — well past
// any joint angle a tessellated rim will hit.
const float MIN_MITER_COS = 0.1;

vec2 to_pixel_space(vec3 world) {
    vec4 clip = u_view_proj * u_model * vec4(world, 1.0);
    vec2 ndc = clip.xy / max(clip.w, 1e-6);
    return ndc * 0.5 * u_viewport_size_px;
}

void main() {
    vec4 this_clip = u_view_proj * u_model * vec4(a_this, 1.0);
    vec2 this_px = this_clip.xy / max(this_clip.w, 1e-6)
                   * 0.5 * u_viewport_size_px;
    vec2 prev_px = to_pixel_space(a_prev);
    vec2 next_px = to_pixel_space(a_next);

    vec2 to_prev = prev_px - this_px;
    vec2 to_next = next_px - this_px;
    float len_in  = length(to_prev);
    float len_out = length(to_next);

    // dir_in: forward direction of the segment arriving at this vertex
    //         (from prev TO this).  dir_out: forward direction leaving.
    // Degenerate prev/next collapse to a single direction → butt cap.
    vec2 dir_in;
    vec2 dir_out;
    bool has_in  = len_in  > 1e-4;
    bool has_out = len_out > 1e-4;
    if (has_in && has_out) {
        dir_in  = -to_prev / len_in;
        dir_out =  to_next / len_out;
    } else if (has_out) {
        dir_out = to_next / len_out;
        dir_in  = dir_out;
    } else if (has_in) {
        dir_in  = -to_prev / len_in;
        dir_out = dir_in;
    } else {
        dir_in  = vec2(1.0, 0.0);
        dir_out = dir_in;
    }

    // Miter length: half_width / cos(angle/2) so the offset corner lies on
    // both adjacent edges' offset lines.  cos(angle/2) == dot(miter_normal,
    // edge_perp) for either neighbouring edge.
    vec2 perp_out = vec2(-dir_out.y, dir_out.x);
    vec2 miter_normal;
    float cos_half;
    if (dot(dir_in, dir_out) < 0.0) {
        // Fold-back joint — e.g. the silhouette vertex of a rim circle
        // viewed edge-on, where the polyline visually reverses direction.
        // The bisector is undefined; the MIN_MITER_COS clamp alone still
        // lets the offset balloon to 1/MIN_MITER_COS of the half-width,
        // which renders as a triangular spike past the rim.  Fall back to
        // a perpendicular butt cap so the stroke terminates cleanly.
        miter_normal = perp_out;
        cos_half = 1.0;
    } else {
        vec2 tangent = dir_in + dir_out;
        float tlen = length(tangent);
        tangent = tlen > 1e-6 ? tangent / tlen : perp_out;
        miter_normal = vec2(-tangent.y, tangent.x);
        cos_half = max(abs(dot(miter_normal, perp_out)), MIN_MITER_COS);
    }
    float half_extent_px = (u_line_half_width_px + AA_EDGE_PX) / cos_half;

    vec2 offset_px = miter_normal * (a_side * half_extent_px);
    vec2 offset_ndc = offset_px / (0.5 * u_viewport_size_px);

    // Reconstruct clip-space position keeping the original z and w so the
    // depth test still works.  A tiny bias toward the camera prevents the
    // stroke from z-fighting with the lens body in the depth pre-pass.
    vec2 this_ndc = this_clip.xy / max(this_clip.w, 1e-6);
    vec2 final_xy = (this_ndc + offset_ndc) * this_clip.w;
    float depth_bias = 1e-3 * this_clip.w;
    gl_Position = vec4(final_xy, this_clip.z - depth_bias, this_clip.w);

    v_world_pos = (u_model * vec4(a_this, 1.0)).xyz;
    // v_side_px is the *unscaled* perpendicular distance — the fragment
    // shader's smoothstep compares it against u_line_half_width_px, so the
    // miter scaling must be undone here, otherwise wider miter corners
    // would read as wider strokes and the fragment AA would underestimate
    // their coverage.
    v_side_px = a_side * (u_line_half_width_px + AA_EDGE_PX);
}
