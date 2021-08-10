#version 300 es
// -----------------------------------------------------------------------------
// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
// -----------------------------------------------------------------------------
precision highp float;

in vec4   a_position;
in vec2   a_texcoord;

out vec2  v_texcoord;
// -----------------------------------------------------------------------------

void main(void) {
  // Some drivers don't like position being written here
  // with the tessellation stages enabled also.
  // Comment next line when Tess.Eval shader is enabled.
  gl_Position = a_position;

  v_texcoord = a_texcoord;
}
