// -----------------------------------------------------------------------------
// Licenses
//  CC0     - https://creativecommons.org/share-your-work/public-domain/cc0/
//  MIT     - https://mit-license.org/
//  WTFPL   - https://en.wikipedia.org/wiki/WTFPL
//  Unknown - No license identified, does not mean public domain
// -----------------------------------------------------------------------------

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
#define BPM         120.0
#define BTIME_1ST   0.0
#define BTIME(n)    (BTIME_1ST+(n)*60.0/BPM)
#define SCA(a)      vec2(sin(a), cos(a))
#define DOT2(x)     dot(x, x)

// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
#define ROTY(a)               \
  mat3(                       \
    +cos(a) , 0.0 , +sin(a)   \
  , 0.0     , 1.0 , 0.0       \
  , -sin(a) , 0.0 , +cos(a)   \
  )

// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
#define ROTZ(a)               \
  mat3(                       \
    +cos(a) , +sin(a) , 0.0   \
  , -sin(a) , +cos(a) , 0.0   \
  , 0.0     , 0.0     , 1.0   \
  )

// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
#define ROTX(a)               \
  mat3(                       \
    1.0 , 0.0     , 0.0       \
  , 0.0 , +cos(a) , +sin(a)   \
  , 0.0 , -sin(a) , +cos(a)   \
  )

// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
mat3 rotX(float a) {
  float c = cos(a);
  float s = sin(a);
  return mat3(
    1.0 , 0.0 , 0.0
  , 0.0 , +c  , +s
  , 0.0 , -s  , +c
  );
}

// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
mat3 rotY(float a) {
  float c = cos(a);
  float s = sin(a);
  return mat3(
    +c  , 0.0 , +s
  , 0.0 , 1.0 , 0.0
  , -s  , 0.0 , +c
  );
}

// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
mat3 rotZ(float a) {
  float c = cos(a);
  float s = sin(a);
  return mat3(
    +c  , +s  , 0.0
  , -s  , +c  , 0.0
  , 0.0 , 0.0 , 1.0
  );
}

// License: MIT, author: Inigo Quilez, found: https://iquilezles.org/articles/noacos/
mat3 rot(vec3 d, vec3 z) {
  vec3  v = cross( z, d );
  float c = dot( z, d );
  float k = 1.0/(1.0+c);

  return mat3( v.x*v.x*k + c,     v.y*v.x*k - v.z,    v.z*v.x*k + v.y,
               v.x*v.y*k + v.z,   v.y*v.y*k + c,      v.z*v.y*k - v.x,
               v.x*v.z*k - v.y,   v.y*v.z*k + v.x,    v.z*v.z*k + c    );
}

// License: CC0, author: blackie, found: https://www.shadertoy.com/view/wtVyWK
vec3 rot(vec3 p, vec3 ax, float ro) {
  return mix(dot(p,ax)*ax,p,cos(ro))+sin(ro)*cross(ax,p);
}


// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
float ref(inout vec3 p, vec3 r) {
  float d = dot(p, r);
  p -= r*min(0.0, d)*2.0;
  return d < 0.0 ? 0.0 : 1.0;
}

// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
float linstep(float edge0, float edge1, float x) {
  return clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
}

// License: Unknown, author: XorDev, found: https://x.com/XorDev/status/1808902860677001297
vec3 hsv2rgb_approx(vec3 hsv) {
  return (cos(hsv.x*2*acos(-1.)+vec3(0,4,2))*hsv.y+2-hsv.y)*hsv.z/2;
}

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

// License: Unknown, author: nmz (twitter: @stormoid), found: https://www.shadertoy.com/view/NdfyRM
vec3 sRGB(vec3 t) {
  return mix(1.055*pow(t, vec3(1./2.4)) - 0.055, 12.92*t, step(t, vec3(0.0031308)));
}

// License: Unknown, author: Matt Taylor (https://github.com/64), found: https://64.github.io/tonemapping/
vec3 aces_approx(vec3 v) {
  const float
    a = 2.51
  , b = 0.03
  , c = 2.43
  , d = 0.59
  , e = 0.14
  ;
  v = max(v, 0.);
  v *= .6;
  return clamp((v*(a*v+b))/(v*(c*v+d)+e), 0., 1.);
}

// License: Unknown, author: nmz (twitter: @stormoid), found: https://www.shadertoy.com/view/NdfyRM
float getsat(vec3 c) {
    float mi = min(min(c.x, c.y), c.z);
    float ma = max(max(c.x, c.y), c.z);
    return (ma - mi)/(ma+ 1e-7);
}

// License: Unknown, author: nmz (twitter: @stormoid), found: https://www.shadertoy.com/view/NdfyRM
#define DSP_STR 1.5
vec3 rgb_lerp(in vec3 a, in vec3 b, in float x) {
    //Interpolated base color (with singularity fix)
    vec3 ic = mix(a, b, x) + vec3(1e-6,0.,0.);

    //Saturation difference from ideal scenario
    float sd = abs(getsat(ic) - mix(getsat(a), getsat(b), x));

    //Displacement direction
    vec3 dir = normalize(vec3(2.*ic.x - ic.y - ic.z, 2.*ic.y - ic.x - ic.z, 2.*ic.z - ic.y - ic.x));
    //Simple Lighntess
    float lgt = dot(vec3(1.0), ic);

    //Extra scaling factor for the displacement
    float ff = dot(dir, normalize(ic));

    //Displace the color
    ic += DSP_STR*dir*sd*ff*lgt;
    return clamp(ic,0.,1.);
}

const mat3
  OKLAB_M1=mat3(
    0.4122214708, 0.5363325363, 0.0514459929
  , 0.2119034982, 0.6806995451, 0.1073969566
  , 0.0883024619, 0.2817188376, 0.6299787005
  )
, OKLAB_M2=mat3(
    0.2104542553,  0.7936177850, -0.0040720468
  , 1.9779984951, -2.4285922050,  0.4505937099
  , 0.0259040371,  0.7827717662, -0.8086757660
  )
, OKLAB_N1 = mat3(
    1,  0.3963377774,  0.2158037573
  , 1, -0.1055613458, -0.0638541728
  , 1, -0.0894841775, -1.2914855480
  )
, OKLAB_N2 = mat3(
     4.0767416621, -3.3077115913,  0.2309699292
  , -1.2684380046,  2.6097574011, -0.3413193965
  , -0.0041960863, -0.7034186147,  1.7076147010
  )
;

// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
#define LINEARTOOKLAB(c) (OKLAB_M2*pow(OKLAB_M1*(c), vec3(1./3.)))
// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
vec3 linearToOklab(vec3 c) {
  return OKLAB_M2*pow(OKLAB_M1*(c), vec3(1./3.));
}

// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
#define OKLABTOLINEAR(c) (OKLAB_N2*pow(OKLAB_N1*(c),vec3(3)))
// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
vec3 oklabToLinear(vec3 c) {
  vec3 v=OKLAB_N1*(c);
  return OKLAB_N2*(v*v*v);
}


// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
vec3  saturate(in vec3 a)   { return clamp(a, 0.0, 1.0); }
vec2  saturate(in vec2 a)   { return clamp(a, 0.0, 1.0); }
float saturate(in float a)  { return clamp(a, 0.0, 1.0); }

// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
float dot2(vec2 x) {
  return dot(x, x);
}


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

// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
vec3 toSpherical(vec3 p) {
  float r = length(p);
  return vec3(r, acos(p.z/r), atan(p.y, p.x));
}

// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
vec3 toRect(vec3 p) {
  float s   = sin(p.y);
  return p.x * vec3(s*cos(p.z), s*sin(p.z), cos(p.y));
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
float modRadial(inout vec2 p, float o, float m) {
  float l = length(p);
  float k = l;
  l -= o;
  float n = mod1(l, m);

  p = (l/k)*p;
  return n;
}

// License: Unknown, author: Martijn Steinrucken, found: https://www.youtube.com/watch?v=VmrIDyYiJBA
vec2 hextile(inout vec2 p) {
  // See Art of Code: Hexagonal Tiling Explained!
  // https://www.youtube.com/watch?v=VmrIDyYiJBA
  const vec2 sz       = vec2(1.0, sqrt(3.0));
  const vec2 hsz      = 0.5*sz;

  vec2 p1 = mod(p, sz)-hsz;
  vec2 p2 = mod(p - hsz, sz)-hsz;
  vec2 p3 = dot(p1, p1) < dot(p2, p2) ? p1 : p2;
  vec2 n = ((p3 - p + hsz)/sz);
  p = p3;

  n -= vec2(0.5);
  // Rounding to make hextile 0,0 well behaved
  return round(n*2.0)*0.5;
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

// License: MIT, author: Inigo Quilez, found: https://iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm
float isosceles(vec2 p, vec2 q) {
  p.x = abs(p.x);
  vec2 a = p - q*clamp(dot(p,q)/dot(q,q), 0.0, 1.0);
  vec2 b = p - q*vec2(clamp(p.x/q.x, 0.0, 1.0), 1.0);
  float s = -sign(q.y);
  vec2 d = min(vec2(dot(a,a), s*(p.x*q.y-p.y*q.x)),
               vec2(dot(b,b), s*(p.y-q.y)));
  return -sqrt(d.x)*sign(d.y);
}

// License: MIT, author: Inigo Quilez, found: https://iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm
float box(vec2 p, vec2 b) {
  vec2 d = abs(p)-b;
  return length(max(d,0.0)) + min(max(d.x,d.y),0.0);
}

// License: MIT, author: Inigo Quilez, found: https://iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm
float hexagon(vec2 p, float r) {
  const vec3 k = 0.5*vec3(-sqrt(3.0), 1.0, sqrt(4.0/3.0));
  p = abs(p);
  p -= 2.0*min(dot(k.xy,p),0.0)*k.xy;
  p -= vec2(clamp(p.x, -k.z*r, k.z*r), r);
  return length(p)*sign(p.y);
}


// License: MIT, author: Inigo Quilez, found: https://iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm
float horseshoe(vec2 p, vec2 c, float r, vec2 w) {
  p.x = abs(p.x);
  float l = length(p);
  p = mat2(-c.x, c.y,
            c.y, c.x)*p;
  p = vec2((p.y>0.0)?p.x:l*sign(-c.x),
           (p.x>0.0)?p.y:l);
  p = vec2(p.x,abs(p.y-r))-w;
  return length(max(p,0.0)) + min(0.0,max(p.x,p.y));
}

// License: MIT, author: Inigo Quilez, found: https://iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm
float segment(vec2 p, vec2 a, vec2 b) {
  vec2 pa = p-a, ba = b-a;
  float h = clamp(dot(pa,ba)/dot(ba,ba), 0.0, 1.0);
  return length(pa - ba*h);
}

// License: MIT, author: Inigo Quilez, found: https://www.shadertoy.com/view/wsGSD3
float snowflake(vec2 p) {
  const vec2 k = vec2(0.5,-sqrt(3.0))/2.0;
  p = p.yx;
  p = abs(p);
  p -= 2.0*min(dot(k,p),0.0)*k;
  p = abs(p);
  float d  = segment(p, vec2(.00, 0), vec2(.75, 0));
  d = min(d, segment(p, vec2(.50, 0), vec2(.50, 0) + .10));
  d = min(d, segment(p, vec2(.25, 0), vec2(.25, 0) + .15));
  return d;
}


// License: MIT, author: Inigo Quilez, found: https://iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm
float parabola(vec2 pos, float k) {
  pos.x = abs(pos.x);
  float ik = 1.0/k;
  float p = ik*(pos.y - 0.5*ik)/3.0;
  float q = 0.25*ik*ik*pos.x;
  float h = q*q - p*p*p;
  float r = sqrt(abs(h));
  float x = (h>0.0) ?
        pow(q+r,1.0/3.0) - pow(abs(q-r),1.0/3.0)*sign(r-q) :
        2.0*cos(atan(r,q)/3.0)*sqrt(p);
  return length(pos-vec2(x,k*x*x)) * sign(pos.x-x);
}

// License: MIT OR CC-BY-NC-4.0, author: mercury, found: https://mercury.sexy/hg_sdf/
float corner(vec2 p) {
  vec2 v = min(p, vec2(0));
  return length(max(p, vec2(0))) + max(v.x, v.y);
}

// License: MIT, author: Inigo Quilez, found: https://iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm
float roundedBox(vec2 p, vec2 b, vec4 r) {
  r.xy = (p.x>0.0)?r.xy : r.zw;
  r.x  = (p.y>0.0)?r.x  : r.y;
  vec2 q = abs(p)-b+r.x;
  return min(max(q.x,q.y),0.0) + length(max(q,0.0)) - r.x;
}

// License: Unknown, author: Unknown, found: shadertoy somewhere, don't remember where
float dfcos(float x) {
  return sqrt(x*x+1.0)*0.8-1.8;
}

// License: Unknown, author: Unknown, found: shadertoy somewhere, don't remember where
float dfcos(vec2 p, float freq) {
  // Approximate distance to cos
  float x = p.x;
  float y = p.y;
  x *= freq;

  float x1 = abs(mod(x+PI,TAU)-PI);
  float x2 = abs(mod(x   ,TAU)-PI);

  float a = 0.18*freq;

  x1 /= max( y*a+1.0-a,1.0);
  x2 /= max(-y*a+1.0-a,1.0);
  return (mix(-dfcos(x2)-1.0,dfcos(x1)+1.0,clamp(y*0.5+0.5,0.0,1.0)))/max(freq*0.8,1.0)+max(abs(y)-1.0,0.0)*sign(y);
}

// License: MIT, author: Inigo Quilez, found: https://iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm
float bezier(vec2 pos, vec2 A, vec2 B, vec2 C) {
  vec2 a = B - A;
  vec2 b = A - 2.0*B + C;
  vec2 c = a * 2.0;
  vec2 d = A - pos;
  float kk = 1.0/dot(b,b);
  float kx = kk * dot(a,b);
  float ky = kk * (2.0*dot(a,a)+dot(d,b)) / 3.0;
  float kz = kk * dot(d,a);
  float res = 0.0;
  float p = ky - kx*kx;
  float p3 = p*p*p;
  float q = kx*(2.0*kx*kx-3.0*ky) + kz;
  float h = q*q + 4.0*p3;
  if(h >= 0.0) {
    h = sqrt(h);
    vec2 x = (vec2(h,-h)-q)/2.0;
    vec2 uv = sign(x)*pow(abs(x), vec2(1.0/3.0));
    float t = clamp(uv.x+uv.y-kx, 0.0, 1.0);
    res = dot2(d + (c + b*t)*t);
  } else {
    float z = sqrt(-p);
    float v = acos(q/(p*z*2.0)) / 3.0;
    float m = cos(v);
    float n = sin(v)*1.732050808;
    vec3  t = clamp(vec3(m+m,-n-m,n-m)*z-kx,0.0,1.0);
    res = min(dot2(d+(c+b*t.x)*t.x),
              dot2(d+(c+b*t.y)*t.y));
    // the third root cannot be the closest
    // res = min(res,dot2(d+(c+b*t.z)*t.z));
  }
  return sqrt(res);
}


// License: MIT, author: Inigo Quilez, found: https://iquilezles.org/articles/intersectors/
vec2 ray_box(vec3 ro, vec3 rd, vec3 boxSize, out vec3 outNormal)  {
  vec3
    m = 1.0/rd  // can precompute if traversing a set of aligned boxes
  , n = m*ro    // can precompute if traversing a set of aligned boxes
  , k = abs(m)*boxSize
  , t1 = -n - k
  , t2 = -n + k
  ;
  float
    tN = max( max( t1.x, t1.y ), t1.z )
  , tF = min( min( t2.x, t2.y ), t2.z )
  ;
  if( tN>tF || tF<0.0) return vec2(MISS); // no intersection
  outNormal = (tN>0.0) ? step(vec3(tN),t1) : // ro ouside the box
                         step(t2,vec3(tF));  // ro inside the box
  outNormal *= -sign(rd);
  return vec2( tN, tF );
}


float ray_xy_plane(vec3 ro, vec3 rd, float o) {
  float t=(o-ro.y)/rd.y;
  return t;
}

vec2 ray_cylinder(vec3 ro, vec3 rd, float ra) {
  float
      a=dot(rd.xy, rd.xy)
    , b=dot(ro.xy, rd.xy)
    , c=dot(ro.xy, ro.xy) - ra*ra
    , h=b*b - a*c
    ;
  if(h < 0.0) return vec2(MISS);
  h = sqrt(h);
  return vec2(-b-h, -b+h)/a;
}

// License: MIT, author: Inigo Quilez, found: https://www.iquilezles.org/www/articles/spherefunctions/spherefunctions.htm
float ray_sphere(vec3 ro, vec3 rd, vec4 sph) {
  vec3 oc = ro - sph.xyz;
  float b = dot(oc, rd);
  float c = dot(oc, oc) - sph.w*sph.w;
  float h = b*b - c;
  if(h<0.0) return -1.0;
  h = sqrt(h);
  return -b - h;
}

// License: MIT, author: Inigo Quilez, found: https://www.iquilezles.org/www/articles/spherefunctions/spherefunctions.htm
vec2 ray_sphere(vec3 ro, vec3 rd, vec4 sph) {
  vec3 oc = ro - sph.xyz;
  float b = dot(oc, rd);
  float c = dot(oc, oc) - sph.w*sph.w;
  float h = b*b - c;
  if(h<0.0) return vec2(-1.0);
  h = sqrt(h);
  return vec2(-b - h, -b + h);
}

// License: MIT, author: Inigo Quilez, found: https://www.iquilezles.org/www/articles/spherefunctions/spherefunctions.htm
float ray_sphere_density(vec3 ro, vec3 rd, vec4 sph, float dbuffer) {
  float ndbuffer = dbuffer/sph.w;
  vec3  rc = (ro - sph.xyz)/sph.w;
  float b = dot(rd,rc);
  float c = dot(rc,rc) - 1.0;
  float h = b*b - c;
  if(h<0.0) return 0.0;
  h = sqrt(h);
  float t1 = -b - h;
  float t2 = -b + h;
  if(t2<0.0 || t1>ndbuffer) return 0.0;
  t1 = max(t1, 0.0);
  t2 = min(t2, ndbuffer);
  float i1 = -(c*t1 + b*t1*t1 + t1*t1*t1/3.0);
  float i2 = -(c*t2 + b*t2*t2 + t2*t2*t2/3.0);
  return (i2-i1)*(3.0/4.0);
}

// License: MIT, author: Inigo Quilez, found: https://www.iquilezles.org/www/articles/intersectors/intersectors.htm
float ray_torus(vec3 ro, vec3 rd, vec2 tor) {
  float po = 1.0;

  float Ra2 = tor.x*tor.x;
  float ra2 = tor.y*tor.y;

  float m = dot(ro,ro);
  float n = dot(ro,rd);

  // bounding sphere
  {
    float h = n*n - m + (tor.x+tor.y)*(tor.x+tor.y);
    if(h<0.0) return -1.0;
    //float t = -n-sqrt(h); // could use this to compute intersections from ro+t*rd
  }

  // find quartic equation
  float k = (m - ra2 - Ra2)/2.0;
  float k3 = n;
  float k2 = n*n + Ra2*rd.z*rd.z + k;
  float k1 = k*n + Ra2*ro.z*rd.z;
  float k0 = k*k + Ra2*ro.z*ro.z - Ra2*ra2;

  #ifndef TORUS_REDUCE_PRECISION
  // prevent |c1| from being too close to zero
  if(abs(k3*(k3*k3 - k2) + k1) < 0.01)
  {
    po = -1.0;
    float tmp=k1; k1=k3; k3=tmp;
    k0 = 1.0/k0;
    k1 = k1*k0;
    k2 = k2*k0;
    k3 = k3*k0;
  }
  #endif

  float c2 = 2.0*k2 - 3.0*k3*k3;
  float c1 = k3*(k3*k3 - k2) + k1;
  float c0 = k3*(k3*(-3.0*k3*k3 + 4.0*k2) - 8.0*k1) + 4.0*k0;


  c2 /= 3.0;
  c1 *= 2.0;
  c0 /= 3.0;

  float Q = c2*c2 + c0;
  float R = 3.0*c0*c2 - c2*c2*c2 - c1*c1;

  float h = R*R - Q*Q*Q;
  float z = 0.0;
  if(h < 0.0) {
    // 4 intersections
    float sQ = sqrt(Q);
    z = 2.0*sQ*cos(acos(R/(sQ*Q)) / 3.0);
  } else {
    // 2 intersections
    float sQ = pow(sqrt(h) + abs(R), 1.0/3.0);
    z = sign(R)*abs(sQ + Q/sQ);
  }
  z = c2 - z;

  float d1 = z   - 3.0*c2;
  float d2 = z*z - 3.0*c0;
  if(abs(d1) < 1.0e-4) {
    if(d2 < 0.0) return -1.0;
    d2 = sqrt(d2);
  } else {
    if(d1 < 0.0) return -1.0;
    d1 = sqrt(d1/2.0);
    d2 = c1/d1;
  }

  //----------------------------------

  float result = 1e20;

  h = d1*d1 - z + d2;
  if(h > 0.0) {
    h = sqrt(h);
    float t1 = -d1 - h - k3; t1 = (po<0.0)?2.0/t1:t1;
    float t2 = -d1 + h - k3; t2 = (po<0.0)?2.0/t2:t2;
    if(t1 > 0.0) result=t1;
    if(t2 > 0.0) result=min(result,t2);
  }

  h = d1*d1 - z - d2;
  if(h > 0.0) {
    h = sqrt(h);
    float t1 = d1 - h - k3;  t1 = (po<0.0)?2.0/t1:t1;
    float t2 = d1 + h - k3;  t2 = (po<0.0)?2.0/t2:t2;
    if(t1 > 0.0) result=min(result,t1);
    if(t2 > 0.0) result=min(result,t2);
  }

  return result;
}

// License: MIT, author: Inigo Quilez, found: https://www.iquilezles.org/www/articles/intersectors/intersectors.htm
vec3 torusNormal(vec3 pos, vec2 tor) {
  return normalize(pos*(dot(pos,pos)- tor.y*tor.y - tor.x*tor.x*vec3(1.0,1.0,-1.0)));
}

// License: MIT, author: Pascal Gilcher, found: https://www.shadertoy.com/view/flSXRV
float atan_approx(float y, float x) {
  float cosatan2 = x / (abs(x) + abs(y));
  float t = PI_2 - cosatan2 * PI_2;
  return y < 0.0 ? -t : t;
}

// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
float acos_approx(float x) {
  return atan_approx(sqrt(max(.0, 1. - x*x)), x);
}

// License: Unknown, author: Claude Brezinski, found: https://mathr.co.uk/blog/2017-09-06_approximating_hyperbolic_tangent.html
float tanh_approx(float x) {
  //  Found this somewhere on the interwebs
  //  return tanh(x);
  float x2 = x*x;
  return clamp(x*(27.0 + x2)/(27.0+9.0*x2), -1.0, 1.0);
}

// License: Unknown, author: XorDev, found: https://bsky.app/profile/xordev.com/post/3m3da656aps2c
float valueNoise(vec2 x)
{
    vec2 i = floor(x);
    vec2 s = smoothstep(i, i+1.0, x);
    return mix(mix(rand(i), rand(i + vec2(1,0)), s.x),
               mix(rand(i+vec2(0,1)), rand(i + 1.0), s.x), s.y);
}

// License: Unknown, author: Unknown, found: don't remember
float hash(float co) {
  return fract(sin(co*12.9898) * 13758.5453);
}

// License: Unknown, author: Unknown, found: don't remember
float hash(vec2 co) {
  return fract(sin(dot(co.xy ,vec2(12.9898,58.233))) * 13758.5453);
}

// License: Unknown, author: Unknown, found: don't remember
vec2 hash2(vec2 p) {
  p = vec2(dot (p, vec2 (127.1, 311.7)), dot (p, vec2 (269.5, 183.3)));
  return fract(sin(p)*43758.5453123);
}

// License: Unknown, author: Unknown, found: don't remember
float hash3(vec3 r)  {
  return fract(sin(dot(r.xy,vec2(1.38984*sin(r.z),1.13233*cos(r.z))))*653758.5453);
}

// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
vec2 shash2(vec2 p) {
  return -1.0+2.0*hash2(p);
}

// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
vec4 alphaBlend(vec4 back, vec4 front) {
  // Based on: https://en.wikipedia.org/wiki/Alpha_compositing
  float w = front.w + back.w*(1.0-front.w);
  vec3 xyz = (front.xyz*front.w + back.xyz*back.w*(1.0-front.w))/w;
  return w > 0.0 ? vec4(xyz, w) : vec4(0.0);
}

// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
vec3 alphaBlend(vec3 back, vec4 front) {
  // Based on: https://en.wikipedia.org/wiki/Alpha_compositing
  return mix(back, front.xyz, front.w);
}

// License: Unknown, author: Shane, found: Discord private message
// Tri-Planar blending function. Based on an old Nvidia writeup:
// GPU Gems 3 - Ryan Geiss: http://http.developer.nvidia.com/GPUGems3/gpugems3_ch01.html
vec3 tex3D(sampler2D tex, vec3 p, vec3 n) {
  n = max(abs(n), 0.001); // n = max((abs(n) - 0.2)*7., 0.001); // n = max(abs(n), 0.001), etc.
  n /= (n.x + n.y + n.z);
  return (texture(tex, p.yz)*n.x + texture(tex, p.zx)*n.y + texture(tex, p.xy)*n.z).xyz;
}

//  simplified version of Dave Hoskins blur
vec3 dblur(vec2 q,float rad) {
  vec3 acc=vec3(0);
  const float m = 0.002;
  vec2 pixel=vec2(m*RESOLUTION.y/RESOLUTION.x,m);
  vec2 angle=vec2(0,rad);
  rad=1.;
  const int iter = 30;
  for (int j=0; j<iter; ++j) {
    rad += 1./rad;
    angle*=brot;
    vec4 col=texture(prevFrame,q+pixel*(rad-1.)*angle);
    acc+=col.xyz;
  }
  return acc*(1.0/float(iter));
}

// License: MIT, author: Inigo Quilez, found: https://www.shadertoy.com/view/XslGRr
float noise(vec2 p) {
  // Found at https://www.shadertoy.com/view/sdlXWX
  // Which then redirected to IQ shader
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f*f*(3.-2.*f);

  float n =
         mix( mix( dot(shash2(i + vec2(0.,0.) ), f - vec2(0.,0.)),
                   dot(shash2(i + vec2(1.,0.) ), f - vec2(1.,0.)), u.x),
              mix( dot(shash2(i + vec2(0.,1.) ), f - vec2(0.,1.)),
                   dot(shash2(i + vec2(1.,1.) ), f - vec2(1.,1.)), u.x), u.y);

  return 2.0*n;
}

// License: MIT, author: Inigo Quilez, found: https://www.shadertoy.com/view/XslGRr
float vnoise(vec2 p) {
   vec2 i = floor( p );
   vec2 f = fract( p );

   vec2 u = f*f*(3.0-2.0*f);

   float a = hash( i + vec2(0.0,0.0) );
   float b = hash( i + vec2(1.0,0.0) );
   float c = hash( i + vec2(0.0,1.0) );
   float d = hash( i + vec2(1.0,1.0) );

   float m0 = mix(a, b, u.x);
   float m1 = mix(c, d, u.x);
   float m2 = mix(m0, m1, u.y);

   return m2;
}

// License: MIT, author: Inigo Quilez, found: https://www.iquilezles.org/www/index.htm
vec3 postProcess(vec3 col, vec2 q) {
  //  Found this somewhere on the interwebs
  col = clamp(col, 0.0, 1.0);
  // Gamma correction
  col = pow(col, 1.0/vec3(2.2));
  col = col*0.6+0.4*col*col*(3.0-2.0*col);
  col = mix(col, vec3(dot(col, vec3(0.33))), -0.4);
  // Vignetting
  col*= 0.5+0.5*pow(19.0*q.x*q.y*(1.0-q.x)*(1.0-q.y),0.7);
  return col;
}

// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
// .yz are the partial derivates of the noise function
vec3 simple_noise(vec2 p) {
  vec2 C=cos(p),S=sin(p);
  return vec3(S.x*S.y,C.x*S.y,S.x*C.y);
}

// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
// .yzw are the partial derivates of the noise function
vec4 simple_noise(vec3 p) {
  vec3 C=cos(p),S=sin(p);
  return vec4(S.x*S.y*S.z,C.x*S.y*S.z,S.x*C.y*S.z,S.x*S.y*C.z);
}

// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
//  Based on simple_dot_noise by XorDev
// .yz are the partial derivates of the noise function
vec3 simple_dot_noise(vec2 p) {
  float noise = dot(sin(p), cos(p*1.618).yx);
  float dx = cos(p.x) * cos(p.y * 1.618) - sin(p.y) * sin(p.x * 1.618) * 1.618;
  float dy = -sin(p.x) * sin(p.y * 1.618) * 1.618 + cos(p.y) * cos(p.x * 1.618);
  return vec3(noise, dx, dy);
}

// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
//  Based on simple_dot_noise by XorDev
// .yzw are the partial derivates of the noise function
vec4 simple_dot_noise(vec3 p) {
  float noise = dot(sin(p), cos(p*1.618).yzx);
  float dx = cos(p.x) * cos(p.y * 1.618) - sin(p.z) * sin(p.x * 1.618) * 1.618;
  float dy = -sin(p.x) * sin(p.y * 1.618) * 1.618 + cos(p.y) * cos(p.z * 1.618);
  float dz = -sin(p.y) * sin(p.z * 1.618) * 1.618 + cos(p.z) * cos(p.x * 1.618);
  return vec4(noise, dx, dy, dz);
}

// License: Unknown, author: XorDev, found: https://www.shadertoy.com/view/wfsyRX
float simple_dot_noise(vec3 p) {
  return dot(sin(p), cos(p*1.618).yzx);
}

// License: Unknown, author: XorDev, found: https://www.shadertoy.com/view/wfsyRX
float dot_noise(vec3 p) {
    //The golden ratio:
    //https://mini.gmshaders.com/p/phi
    const float phi = 1.618033988;
    //Rotating the golden angle on the vec3(1, phi, phi*phi) axis
    const mat3 gold = mat3(
    -0.571464913, +0.814921382, +0.096597072,
    -0.278044873, -0.303026659, +0.911518454,
    +0.772087367, +0.494042493, +0.399753815);

    //Gyroid with irrational orientations and scales
    return dot(cos(gold * p), sin(phi * p * gold));
    //Ranges from [-3 to +3]
}

const vec3
  lum_weights_linear = vec3(0.2126, 0.7152, 0.0722)
, lum_weights_srgb   = vec3(0.299, 0.587, 0.114)
;

// License: Unknown, author: XorDev, found: https://github.com/XorDev/GM_FXAA
vec4 fxaa(sampler2D tex, vec2 uv, vec2 texelSz) {
  // See this blog
  // https://mini.gmshaders.com/p/gm-shaders-mini-fxaa

  // Maximum texel span
  const float span_max    = 8.0;
  // These are more technnical and probably don't need changing:
  // Minimum "dir" reciprocal
  const float reduce_min  = (1.0/128.0);
  // Luma multiplier for "dir" reciprocal
  const float reduce_mul  = (1.0/32.0);

  const vec3  luma        = vec3(0.299, 0.587, 0.114);

  // Sample center and 4 corners
  vec3 rgbCC = texture(tex, uv).rgb;
  vec3 rgb00 = texture(tex, uv+vec2(-0.5,-0.5)*texelSz).rgb;
  vec3 rgb10 = texture(tex, uv+vec2(+0.5,-0.5)*texelSz).rgb;
  vec3 rgb01 = texture(tex, uv+vec2(-0.5,+0.5)*texelSz).rgb;
  vec3 rgb11 = texture(tex, uv+vec2(+0.5,+0.5)*texelSz).rgb;

  //Get luma from the 5 samples
  float lumaCC = dot(rgbCC, luma);
  float luma00 = dot(rgb00, luma);
  float luma10 = dot(rgb10, luma);
  float luma01 = dot(rgb01, luma);
  float luma11 = dot(rgb11, luma);

  // Compute gradient from luma values
  vec2 dir = vec2((luma01 + luma11) - (luma00 + luma10), (luma00 + luma01) - (luma10 + luma11));

  // Diminish dir length based on total luma
  float dirReduce = max((luma00 + luma10 + luma01 + luma11) * reduce_mul, reduce_min);

  // Divide dir by the distance to nearest edge plus dirReduce
  float rcpDir = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);

  // Multiply by reciprocal and limit to pixel span
  dir = clamp(dir * rcpDir, -span_max, span_max) * texelSz.xy;

  // Average middle texels along dir line
  vec4 A = 0.5 * (
      texture2D(tex, uv - dir * (1.0/6.0))
    + texture2D(tex, uv + dir * (1.0/6.0))
    );

  // Average with outer texels along dir line
  vec4 B = A * 0.5 + 0.25 * (
      texture2D(tex, uv - dir * (0.5))
    + texture2D(tex, uv + dir * (0.5))
    );


  // Get lowest and highest luma values
  float lumaMin = min(lumaCC, min(min(luma00, luma10), min(luma01, luma11)));
  float lumaMax = max(lumaCC, max(max(luma00, luma10), max(luma01, luma11)));

  // Get average luma
  float lumaB = dot(B.rgb, luma);

  //If the average is outside the luma range, using the middle average
  return ((lumaB < lumaMin) || (lumaB > lumaMax)) ? A : B;
}

// License: MIT, author: Inigo Quilez, found: https://www.shadertoy.com/view/ttcyRS
vec3 oklab_mix(vec3 lin1, vec3 lin2, float a) {
    // https://bottosson.github.io/posts/oklab
    const mat3 kCONEtoLMS = mat3(
         0.4121656120,  0.2118591070,  0.0883097947,
         0.5362752080,  0.6807189584,  0.2818474174,
         0.0514575653,  0.1074065790,  0.6302613616);
    const mat3 kLMStoCONE = mat3(
         4.0767245293, -1.2681437731, -0.0041119885,
        -3.3072168827,  2.6093323231, -0.7034763098,
         0.2307590544, -0.3411344290,  1.7068625689);

    // rgb to cone (arg of pow can't be negative)
    vec3 lms1 = pow( kCONEtoLMS*lin1, vec3(1.0/3.0) );
    vec3 lms2 = pow( kCONEtoLMS*lin2, vec3(1.0/3.0) );
    // lerp
    vec3 lms = mix( lms1, lms2, a );
    // gain in the middle (no oklab anymore, but looks better?)
    lms *= 1.0+0.2*a*(1.0-a);
    // cone to rgb
    return kLMStoCONE*(lms*lms*lms);
}

mat3 fancyRotation(float time) {
  float
    angle1 = time * 0.5
  , angle2 = time * 0.707
  , angle3 = time * 0.33
  , c1 = cos(angle1); float s1 = sin(angle1)
  , c2 = cos(angle2); float s2 = sin(angle2)
  , c3 = cos(angle3); float s3 = sin(angle3)
  ;

  return mat3(
      c1 * c2,
      c1 * s2 * s3 - c3 * s1,
      s1 * s3 + c1 * c3 * s2,

      c2 * s1,
      c1 * c3 + s1 * s2 * s3,
      c3 * s1 * s2 - c1 * s3,

      -s2,
      c2 * s3,
      c2 * c3
  );
}


// Create a quaternion from axis and angle
vec4 createQuaternion(vec3 axis, float angle) {
  float halfAngle = angle * 0.5;
  float s = sin(halfAngle);
  return vec4(axis * s, cos(halfAngle));
}

// Quaternion multiplication
vec4 multiplyQuaternions(vec4 q1, vec4 q2) {
  return vec4(
    q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
    q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
    q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w,
    q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z
  );
}

// Rotate a vector using a quaternion
mat3 rotationFromQuaternion(vec4 q) {
  // Convert quaternion to a rotation matrix
  mat3 rotationMatrix = mat3(
    1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    2.0 * (q.x * q.y - q.w * q.z),
    2.0 * (q.x * q.z + q.w * q.y),

    2.0 * (q.x * q.y + q.w * q.z),
    1.0 - 2.0 * (q.x * q.x + q.z * q.z),
    2.0 * (q.y * q.z - q.w * q.x),

    2.0 * (q.x * q.z - q.w * q.y),
    2.0 * (q.y * q.z + q.w * q.x),
    1.0 - 2.0 * (q.x * q.x + q.y * q.y)
  );

  return rotationMatrix;
}

