#version 330 core

in vec2 v_uv;

layout(location = 0) out vec4 fragColor;

uniform vec3 u_top;
uniform vec3 u_bottom;
// Camera basis vectors in world space.  u_cam_forward is the direction the
// camera looks (world-space); u_cam_right / u_cam_up span the screen plane.
// For each pixel we synthesise a "view ray" in world space:
//
//     dir = normalize(forward + ndc.x*aspect*fov*right + ndc.y*fov*up)
//
// and then map its world Y to the gradient.  This makes the background act
// like a coarse directional skybox: looking down (-Y) -> mostly dark,
// looking up (+Y) -> mostly bright, side-on -> vertical gradient as before.
// The camera is orthographic so all scene rays are actually parallel; the
// faux-FOV here exists only to give the gradient a slight spherical curve
// across the frame so the viewer can tell where they're pointing.
uniform vec3 u_cam_forward;
uniform vec3 u_cam_right;
uniform vec3 u_cam_up;
uniform float u_sky_fov_scale;
uniform float u_aspect;

void main() {
    vec2 ndc = v_uv * 2.0 - 1.0;
    vec2 sample_xy = vec2(ndc.x * u_aspect, ndc.y) * u_sky_fov_scale;
    vec3 dir = normalize(
        u_cam_forward
        + sample_xy.x * u_cam_right
        + sample_xy.y * u_cam_up
    );
    float t = clamp(0.5 + 0.5 * dir.y, 0.0, 1.0);
    vec3 col = mix(u_bottom, u_top, t);
    fragColor = vec4(col, 1.0);
}
