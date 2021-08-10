// Licenses
//  CC0   - https://creativecommons.org/share-your-work/public-domain/cc0/
//  MIT   - https://mit-license.org/
//  WTFPL - https://en.wikipedia.org/wiki/WTFPL

// All my code is licensed under CC0 which relinquishes copyright.
//  No need to reference me or keep the copyright license notification

// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
#define TIME        time
#define TTIME       (TAU*TIME)
#define RESOLUTION  resolution
#define PI          3.141592654
#define PI_2        (0.5*PI)
#define PI_4        (0.25*PI)
#define TAU         (2.0*PI)
#define ROT(a)      mat2(cos(a), sin(a), -sin(a), cos(a))
#define PSIN(x)     (0.5+0.5*sin(x))
#define PCOS(x)     (0.5+0.5*cos(x))

// License: WTFPL, author: sam hocevar, found: https://stackoverflow.com/a/17897228/418488
const vec4 hsv2rgb_K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
vec3 hsv2rgb(vec3 c) {
  vec3 p = abs(fract(c.xxx + hsv2rgb_K.xyz) * 6.0 - hsv2rgb_K.www);
  return c.z * mix(hsv2rgb_K.xxx, clamp(p - hsv2rgb_K.xxx, 0.0, 1.0), c.y);
}
// License: WTFPL, author: sam hocevar, found: https://stackoverflow.com/a/17897228/418488
//  Macro version of above to enable compile-time constants
#define HSV2RGB(c)  (c.z * mix(hsv2rgb_K.xxx, clamp(abs(fract(c.xxx + hsv2rgb_K.xyz) * 6.0 - hsv2rgb_K.www) - hsv2rgb_K.xxx, 0.0, 1.0), c.y))

// License: WTFPL, author: sam hocevar, found: https://stackoverflow.com/a/17897228/418488
vec3 rgb2hsv(vec3 c) {
  const vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
  vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
  vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

  float d = q.x - min(q.w, q.y);
  float e = 1.0e-10;
  return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
vec3  saturate(in vec3 a)   { return clamp(a, 0.0, 1.0); }
vec2  saturate(in vec2 a)   { return clamp(a, 0.0, 1.0); }
float saturate(in float a)  { return clamp(a, 0.0, 1.0); }

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
float pabs(float a, float k) {
  return pmax(a, -a, k);
}

// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
vec2 toPolar(vec2 p) {
  return vec2(length(p), atan(p.y, p.x));
}

// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
vec2 toRect(vec2 p) {
  return vec2(p.x*cos(p.y), p.x*sin(p.y));
}

// License: MIT OR CC-BY-NC-4.0, author: mercury, found: https://mercury.sexy/hg_sdf/
float mod1(inout float p, float size) {
  float halfsize = size*0.5;
  float c = floor((p + halfsize)/size);
  p = mod(p + halfsize, size) - halfsize;
  return c;
}

// License: MIT OR CC-BY-NC-4.0, author: mercury, found: https://mercury.sexy/hg_sdf/
vec2 mod2(inout vec2 p, vec2 size) {
  vec2 c = floor((p + size*0.5)/size);
  p = mod(p + size*0.5,size) - size*0.5;
  return c;
}

// License: MIT OR CC-BY-NC-4.0, author: mercury, found: https://mercury.sexy/hg_sdf/
float modMirror1(inout float p, float size) {
  float halfsize = size*0.5;
  float c = floor((p + halfsize)/size);
  p = mod(p + halfsize,size) - halfsize;
  p *= mod(c, 2.0)*2.0 - 1.0;
  return c;
}

// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
float smoothKaleidoscope(inout vec2 p, float sm, float rep) {
  vec2 hp = p;

  vec2 hpp = toPolar(hp);
  float rn = modMirror1(hpp.y, TAU/rep);

  float sa = PI/rep - pabs(PI/rep - abs(hpp.y), sm);
  hpp.y = sign(hpp.y)*(sa);

  hp = toRect(hpp);

  p = hp;

  return rn;
}

// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
float circle(vec2 p, float r) {
  return length(p) - r;
}

// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
float vesica(vec2 p, vec2 sz) {
  if (sz.x < sz.y) {
    sz = sz.yx;
  } else {
    p  = p.yx;
  }
  vec2 sz2 = sz*sz;
  float d  = (sz2.x-sz2.y)/(2.0*sz.y);
  float r  = sqrt(sz2.x+d*d);
  float b  = sz.x;
  p = abs(p);
  return ((p.y-b)*d>p.x*b) ? length(p-vec2(0.0,b))
                           : length(p-vec2(-d,0.0))-r;
}

// License: MIT, author: Inigo Quilez, found: https://iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm
float box(vec2 p, vec2 b) {
  vec2 d = abs(p)-b;
  return length(max(d,0.0)) + min(max(d.x,d.y),0.0);
}

// License: MIT, author: Inigo Quilez, found: https://www.iquilezles.org/www/articles/spherefunctions/spherefunctions.htm
float raySphere(vec3 ro, vec3 rd, vec4 sph) {
    vec3 oc = ro - sph.xyz;
    float b = dot( oc, rd );
    float c = dot( oc, oc ) - sph.w*sph.w;
    float h = b*b - c;
    if( h<0.0 ) return -1.0;
    h = sqrt( h );
    return -b - h;
}

// License: MIT, author: Pascal Gilcher, found: https://www.shadertoy.com/view/flSXRV
float atan_approx(float y, float x) {
  float cosatan2 = x / (abs(x) + abs(y));
  float t = PI_2 - cosatan2 * PI_2;
  return y < 0.0 ? -t : t;
}

// License: Unknown, author: Unknown, found: don't remember
//  Found this somewhere on the interwebs
float tanh_approx(float x) {
//  return tanh(x);
  float x2 = x*x;
  return clamp(x*(27.0 + x2)/(27.0+9.0*x2), -1.0, 1.0);
}

// License: Unknown, author: Unknown, found: don't remember
float hash(float co) {
  return fract(sin(co*12.9898) * 13758.5453);
}

// License: Unknown, author: Unknown, found: don't remember
vec2 hash(vec2 p) {
  p = vec2(dot (p, vec2 (127.1, 311.7)), dot (p, vec2 (269.5, 183.3)));
  return -1. + 2.*fract (sin (p)*43758.5453123);
}

// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
vec4 alphaBlend(vec4 back, vec4 front) {
  float w = front.w + back.w*(1.0-front.w);
  vec3 xyz = (front.xyz*front.w + back.xyz*back.w*(1.0-front.w))/w;
  return w > 0.0 ? vec4(xyz, w) : vec4(0.0);
}

// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
vec3 alphaBlend(vec3 back, vec4 front) {
  return mix(back, front.xyz, front.w);
}

// License: Unknown, author: Unknown, found: don't remember
//  Found this somewhere on the interwebs
vec3 postProcess(vec3 col, vec2 q) {
  col = clamp(col, 0.0, 1.0);
  col = pow(col, 1.0/vec3(2.2));
  col = col*0.6+0.4*col*col*(3.0-2.0*col);
  col = mix(col, vec3(dot(col, vec3(0.33))), -0.4);
  col *=0.5+0.5*pow(19.0*q.x*q.y*(1.0-q.x)*(1.0-q.y),0.7);
  return col;
}
