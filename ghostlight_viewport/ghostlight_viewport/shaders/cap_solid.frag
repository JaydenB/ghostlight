#version 330 core

// Opaque cross-section cap fill.  Runs in a dedicated post-composite pass
// (default FBO, standard alpha blend, stencil-masked to the cut polygon) so
// the cap reads as flat solid grey instead of going through the OIT
// accumulator with everything else.  Without this, even at α=4.0 the cap
// composites against the visible wall fragments behind it and never reaches
// the apparent opacity of the wall collar between two surfaces.

in vec3 v_world_pos;
out vec4 fragColor;

uniform vec4 u_other_plane;   // second clip plane, used to trim the cap; zero = inactive
uniform vec3 u_color;

void main() {
    if (dot(u_other_plane.xyz, u_other_plane.xyz) > 0.5) {
        if (dot(v_world_pos, u_other_plane.xyz) + u_other_plane.w > 0.0) {
            discard;
        }
    }
    fragColor = vec4(u_color, 1.0);
}
