#version 330 core

uniform mat4 u_view_proj;
uniform vec3 u_plane_center;     // a point on the clip plane
uniform vec3 u_plane_tangent;    // unit basis vector lying in the plane
uniform vec3 u_plane_bitangent;  // unit basis vector lying in the plane, perp to tangent
uniform float u_half_extent;     // half-size of the cap quad in plane units

out vec3 v_world_pos;

void main() {
    vec2 corners[6] = vec2[6](
        vec2(-1.0, -1.0),
        vec2( 1.0, -1.0),
        vec2( 1.0,  1.0),
        vec2(-1.0, -1.0),
        vec2( 1.0,  1.0),
        vec2(-1.0,  1.0)
    );
    vec2 uv = corners[gl_VertexID] * u_half_extent;
    vec3 world = u_plane_center
               + uv.x * u_plane_tangent
               + uv.y * u_plane_bitangent;
    v_world_pos = world;
    gl_Position = u_view_proj * vec4(world, 1.0);
}
