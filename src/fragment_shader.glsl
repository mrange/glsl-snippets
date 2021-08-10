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

void main(void) {
  vec2 q = v_texcoord;
  vec2 p = -1. + 2. * q;
  p.x *= RESOLUTION.x/RESOLUTION.y;

  vec3 col = vec3(0.1);

  fragColor = vec4(col, 1.0);
}
