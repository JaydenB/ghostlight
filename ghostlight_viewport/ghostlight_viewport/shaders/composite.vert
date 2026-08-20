#version 330 core

// Fullscreen triangle generated entirely from gl_VertexID — caller binds an
// empty VAO and issues glDrawArrays(GL_TRIANGLES, 0, 3).  Using a single
// oversized triangle avoids the diagonal seam two-triangle quads can produce
// when texelFetch lands on the seam at certain framebuffer sizes.

out vec2 v_uv;

const vec2 verts[3] = vec2[3](
    vec2(-1.0, -1.0),
    vec2( 3.0, -1.0),
    vec2(-1.0,  3.0)
);

void main() {
    vec2 p = verts[gl_VertexID];
    v_uv = p * 0.5 + 0.5;
    gl_Position = vec4(p, 0.0, 1.0);
}
