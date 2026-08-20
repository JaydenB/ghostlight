#version 330 core

in vec3 v_world_pos;
in float v_kind;

out vec4 fragColor;

uniform vec3 u_tint;
uniform vec3 u_wall_tint;
uniform float u_alpha;       // cap (optical-surface) alpha
uniform float u_wall_alpha;  // side-wall alpha (caller drives this independently
                             // so walls can read more opaque than the tinted caps)
uniform int u_unlit;
uniform vec4 u_clip_plane;
uniform vec4 u_clip_plane_b;

void main() {
    if (dot(u_clip_plane.xyz, u_clip_plane.xyz) > 0.5) {
        if (dot(v_world_pos, u_clip_plane.xyz) + u_clip_plane.w > 0.0) {
            discard;
        }
    }
    if (dot(u_clip_plane_b.xyz, u_clip_plane_b.xyz) > 0.5) {
        if (dot(v_world_pos, u_clip_plane_b.xyz) + u_clip_plane_b.w > 0.0) {
            discard;
        }
    }

    if (u_unlit != 0) {
        // Picking + stencil + cap-quad-stencil-build passes write the
        // tint/alpha straight to the pixel — the pick FBO reads them
        // back as the encoded element/surface id, so wall-tint mixing
        // must NOT apply here, and the OIT premultiplication below would
        // corrupt the id encoding.
        fragColor = vec4(u_tint, u_alpha);
        return;
    }

    // Normal translucent lens render — additive OIT.  The caller has bound
    // an RGBA16F accumulator with GL_ONE, GL_ONE blending, so every
    // fragment that survives the clip discards above adds
    //   (color * alpha, alpha)
    // to that buffer.  composite.frag later turns the accumulator into
    // an alpha-weighted average color + saturating visibility and blends
    // the result onto the screen.  Pre-multiplying the colour by alpha
    // here is what makes ``accum.rgb / accum.a`` the right per-pixel
    // weighted average (heavier-α fragments pull the colour toward
    // themselves) instead of a flat unweighted mix.
    float wall = clamp(v_kind, 0.0, 1.0);
    vec3 color = mix(u_tint, u_wall_tint, wall);
    float alpha = mix(u_alpha, u_wall_alpha, wall);
    fragColor = vec4(color * alpha, alpha);
}
