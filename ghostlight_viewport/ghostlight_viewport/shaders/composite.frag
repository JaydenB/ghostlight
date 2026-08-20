#version 330 core

// Composite pass for additive-accumulator OIT.
//
// During the lens pass, every fragment writes
//   accum += vec4(rgb * alpha, alpha)
// into an RGBA16F target with GL_ONE, GL_ONE blending.  Because the blend
// is commutative, fragments from any number of overlapping translucent
// surfaces contribute regardless of the order they were drawn — that's
// what makes the result per-pixel correct instead of per-primitive.
//
// At composite time we reconstruct:
//   avgColor   = accum.rgb / accum.a    (alpha-weighted color average)
//   visibility = 1 - exp(-accum.a)      (approx. 1 - Π(1-αᵢ); under-estimates
//                                        slightly at high coverage but is
//                                        monotone and saturates to 1)
// then blend the result onto the background with standard
// GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA so the background still shows
// through wherever the lens didn't cover the pixel.

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_accum;

void main() {
    vec4 accum = texture(u_accum, v_uv);
    if (accum.a < 1e-5) discard;
    vec3 avg_color = accum.rgb / accum.a;
    float visibility = 1.0 - exp(-accum.a);
    fragColor = vec4(avg_color, visibility);
}
