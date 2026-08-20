#version 330 core
// Hardcoded fullscreen-triangle pattern -- no attribute input needed.
out vec2 v_uv;
void main() {
    vec2 verts[3] = vec2[3](
        vec2(-1.0, -1.0),
        vec2( 3.0, -1.0),
        vec2(-1.0,  3.0)
    );
    vec2 p = verts[gl_VertexID];
    v_uv = (p + 1.0) * 0.5;
    gl_Position = vec4(p, 0.0, 1.0);
}
