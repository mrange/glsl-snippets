// -----------------------------------------------------------------------------
// Licenses
//  CC0     - https://creativecommons.org/share-your-work/public-domain/cc0/
//  MIT     - https://mit-license.org/
//  WTFPL   - https://en.wikipedia.org/wiki/WTFPL
//  Unknown - No license identified, does not mean public domain
// -----------------------------------------------------------------------------

// Impulse distance field font

// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
#define PI              3.141592654
#define ROT(a)          mat2(cos(a), sin(a), -sin(a), cos(a))
#define SCA(a)          vec2(sin(a), cos(a))

const vec2 impulse_sca0 = SCA(0.0);

// License: MIT, author: Inigo Quilez, found: https://www.iquilezles.org/www/articles/smin/smin.htm
float pmin(float a, float b, float k) {
  float h = clamp(0.5+0.5*(b-a)/k, 0.0, 1.0);
  return mix(b, a, h) - k*h*(1.0-h);
}

// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
float pmax(float a, float b, float k) {
  return -pmin(-a, -b, k);
}

// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
float circle(vec2 p, float r) {
  return length(p) - r;
}

// License: MIT, author: Inigo Quilez, found: https://www.iquilezles.org/www/articles/smin/smin.htm
float box(vec2 p, vec2 b) {
  vec2 d = abs(p)-b;
  return length(max(d,0.0)) + min(max(d.x,d.y),0.0);
}

// License: MIT, author: Inigo Quilez, found: https://www.iquilezles.org/www/articles/smin/smin.htm
float horseshoe(vec2 p, vec2 c, float r, vec2 w) {
  p.x = abs(p.x);
  float l = length(p);
  p = mat2(-c.x, c.y,
            c.y, c.x)*p;
  p = vec2((p.y>0.0)?p.x:l*sign(-c.x),
            (p.x>0.0)?p.y:l );
  p = vec2(p.x,abs(p.y-r))-w;
  return length(max(p,0.0)) + min(0.0,max(p.x,p.y));
}

// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
float impulse_e(inout vec2 pp, float off) {
  vec2 p = pp;
  pp.x -= 1.05+off;
  p -= vec2(0.5, 0.5);
  // TODO: Optimize
  return min(box(p, vec2(0.4, 0.1)), max(circle(p, 0.5), -circle(p, 0.3)));
}

// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
float impulse_I(inout vec2 pp, float off) {
  vec2 p = pp;
  pp.x -= 0.25+off;
  p -= vec2(0.125, 0.75);
  return box(p, vec2(0.125, 0.75));
}

// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
float impulse_l(inout vec2 pp, float off) {
  vec2 p = pp;
  pp.x -= 0.2+off;
  p -= vec2(0.10, 0.5);
  return box(p, vec2(0.1, 0.666));
}

// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
float impulse_m(inout vec2 pp, float off) {
  vec2 p = pp;
  pp.x -= 2.2+off;
  p -= vec2(1.1, 0.5);
  p.y = -p.y;
  p.x = abs(p.x);
  p -= vec2(0.5, 0.0);
  float d = horseshoe(p, impulse_sca0, 0.5, vec2(0.5, 0.1));
  return d;
}

// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
float impulse_n(inout vec2 pp, float off) {
  vec2 p = pp;
  pp.x -= 1.15+off;
  p -= vec2(0.55, 0.5);
  p.y = -p.y;
  float l = horseshoe(p, impulse_sca0, 0.5, vec2(0.5, 0.1));
  return l;
}

// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
float impulse_p(inout vec2 pp, float off) {
  vec2 p = pp;
  pp.x -= 1.05+off;
  p -= vec2(0.55, 0.5);
  float b = box(p - vec2(-0.45, -0.25), vec2(0.1, 0.75));
  float c = abs(circle(p, 0.4)) - 0.1;
  return min(b, c);
}

// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
float impulse_r(inout vec2 pp, float off) {
  vec2 p = pp;
  pp.x -= 0.6+off;
  p -= vec2(0.1, 0.5);
  float d0 = box(p-vec2(0.20, 0.4), vec2(0.3, 0.1));
  float d1 = box(p, vec2(0.1, 0.5));
  return min(d0, d1);
}

// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
float impulse_s(inout vec2 pp, float off) {
  const mat2 rots1 = ROT(-PI/6.0-PI/2.0);
  const mat2 rots2 = ROT(PI);
  vec2 p = pp;
  pp.x -= 0.875+off;
  p -= vec2(0.435, 0.5);
  p *= rots1;
  float u = horseshoe(p - vec2(-0.25*3.0/4.0, -0.125/2.0), impulse_sca0, 0.375, vec2(0.2, 0.1));
  p *= rots2;
  float l = horseshoe(p - vec2(-0.25*3.0/4.0, -0.125/2.0), impulse_sca0, 0.375, vec2(0.2, 0.1));
  return min(u,l);
}

// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
float impulse_u(inout vec2 pp, float off) {
  vec2 p = pp;
  pp.x -= 1.2+off;
  p -= vec2(0.6, 0.475);
  return horseshoe(p - vec2(0.0, 0.125), impulse_sca0, 0.5, vec2(0.4, 0.1));
}

// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
float impulse_t(inout vec2 pp, float off) {
  vec2 p = pp;
  pp.x -= 0.6+off;
  p -= vec2(0.3, 0.6);
  float d0 = box(p-vec2(0.0, 0.3), vec2(0.3, 0.1));
  float d1 = box(p, vec2(0.1, 0.6));
  return min(d0, d1);
}

// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
float impulse(vec2 p, float off) {
  p += vec2(3.385+3.0*off, 0.5);

  float d = 1E6;
  d = min(d, impulse_I(p, off));
  d = min(d, impulse_m(p, off));
  d = min(d, impulse_p(p, off));
  d = min(d, impulse_u(p, off));
  d = min(d, impulse_l(p, off));
  d = min(d, impulse_s(p, off));
  d = min(d, impulse_e(p, off));

  return d;
}

// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
float presents(vec2 p, float off) {
  p += vec2(3.65+3.5*off, 0.5);

  float d = 1E6;
  d = min(d, impulse_p(p, off));
  d = min(d, impulse_r(p, off));
  d = min(d, impulse_e(p, off));
  d = min(d, impulse_s(p, off));
  d = min(d, impulse_e(p, off));
  d = min(d, impulse_n(p, off));
  d = min(d, impulse_t(p, off));
  d = min(d, impulse_s(p, off));

  return d;
}

// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
float impulse_bars(vec2 p, float d) {
  float db = abs(abs(p.y)-0.1)-0.05;
  return pmax(d, -db, 0.025);
}

// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
float impulse(vec2 p) {
  float d = impulse(p, 0.25);
  return impulse_bars(p, d);
}

// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
float presents(vec2 p) {
  float d = presents(p, 0.25);
  return impulse_bars(p, d);
}

