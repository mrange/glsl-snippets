#version 300 es
// -----------------------------------------------------------------------------
// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
// -----------------------------------------------------------------------------
precision highp float;
uniform float time;
uniform vec2 resolution;
in vec2 v_texcoord;
out vec4 fragColor;

// -----------------------------------------------------------------------------

#define TIME        time
#define RESOLUTION  resolution

float circle(vec2 p, float r) {
  return length(p) - r;
}

void main(void) {
  vec2 q = v_texcoord;
  vec2 p = -1. + 2. * q;
  p.x *= RESOLUTION.x/RESOLUTION.y;
  float aa = 2.0/RESOLUTION.y;

  float d = circle(p-0.5*vec2(sin(TIME), sin(TIME*sqrt(0.5))), 0.5);

  vec3 col = vec3(0.1);
  col = mix(col, vec3(1.0), smoothstep(aa, -aa, d));

  fragColor = vec4(col, 1.0);
}
