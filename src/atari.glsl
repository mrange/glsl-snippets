// -----------------------------------------------------------------------------
// Licenses referenced:
//  CC0   - https://creativecommons.org/share-your-work/public-domain/cc0/
//  MIT   - https://mit-license.org/
//  WTFPL - https://en.wikipedia.org/wiki/WTFPL
// -----------------------------------------------------------------------------

// Atari logo and text

// License: MIT, author: Inigo Quilez, found: https://www.iquilezles.org/www/articles/smin/smin.htm
float pmin(float a, float b, float k) {
  float h = clamp(0.5+0.5*(b-a)/k, 0.0, 1.0);
  return mix(b, a, h) - k*h*(1.0-h);
}

float pmax(float a, float b, float k) {
  return -pmin(-a, -b, k);
}

float pabs(float a, float k) {
  return pmax(a, -a, k);
}

// License: MIT, author: Inigo Quilez, found: https://iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm
float box(vec2 p, vec2 b) {
  vec2 d = abs(p)-b;
  return length(max(d,0.0)) + min(max(d.x,d.y),0.0);
}

float circle(vec2 p, float r) {
  return length(p) - r;
}

// License: MIT, author: Inigo Quilez, found: https://iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm
float isosceles(vec2 p, vec2 q) {
  p.x = abs(p.x);
  vec2 a = p - q*clamp( dot(p,q)/dot(q,q), 0.0, 1.0 );
  vec2 b = p - q*vec2( clamp( p.x/q.x, 0.0, 1.0 ), 1.0 );
  float s = -sign( q.y );
  vec2 d = min( vec2( dot(a,a), s*(p.x*q.y-p.y*q.x) ),
                vec2( dot(b,b), s*(p.y-q.y)  ));
  return -sqrt(d.x)*sign(d.y);
}

// License: MIT, author: Inigo Quilez, found: https://iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm
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

// License: MIT, author: Inigo Quilez, found: https://iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm
float segment(vec2 p, vec2 a, vec2 b) {
  vec2 pa = p-a, ba = b-a;
  float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
  return length( pa - ba*h );
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

float atari_logo(vec2 p) {
  p.x = abs(p.x);
  float db = box(p, vec2(0.36, 0.32));

  float dp0 = -parabola(p-vec2(0.4, -0.235), 4.0);
  float dy0 = p.x-0.115;
  float d0 = mix(dp0, dy0, smoothstep(-0.25, 0.125, p.y)); // Very hacky

  float dp1 = -parabola(p-vec2(0.4, -0.32), 3.0);
  float dy1 = p.x-0.07;
  float d1 = mix(dp1, dy1, smoothstep(-0.39, 0.085, p.y)); // Very hacky

  float d2 = p.x-0.035;
  const float sm = 0.025;
  float d = 1E6;
  d = min(d, max(d0, -d1));;
  d = pmin(d, d2, sm);
  d = pmax(d, db, sm);

  return d;
}

float atari_a(inout vec2 p, vec2 off) {
  p -= vec2(0.275, 0.0);

  float d0 = isosceles(p*vec2(1.0, -1.0)-vec2(0.0, -0.225), vec2(0.20, 0.65))-0.1;
  float d1 = isosceles(p*vec2(1.0, -1.0)-vec2(0.0, -0.18), vec2(0.13, 0.55))-0.005;
  float d2 = box(p-vec2(0.0, -0.135), vec2(0.15, 0.06));
  float d3 = p.y+0.325;

  float d = d0;
  d = max(d, -d1);
  d = pmin(d, d2, 0.0125);
  d = pmax(d, -d3, 0.0125);

  p -= vec2(0.275, 0.0) + off;

  return d;
}

float atari_i(inout vec2 p, vec2 off) {
  p -= vec2(0.07, 0.0);

  float d0 = box(p, vec2(0.07, 0.325)-0.0125)-0.0125;

  float d = d0;

  p -= vec2(0.07, 0.0) + off;
  return d;
}

float atari_r(inout vec2 p, vec2 off) {
  p -= vec2(0.22, 0.0);

  float d0 = p.y+0.325;
  float d1 = circle(p - vec2(-0.12, 0.225), 0.1);
  const float a = PI/2.0;
  const vec2 c = vec2(cos(a), sin(a));
  vec2 hp = p;
  hp -= vec2(0.0, 0.14);
  hp.xy = -hp.yx;
  float d2 = horseshoe(hp, c, 0.125, 0.2175*vec2(0.12,0.275));
  float d3 = segment(p-vec2(-0.015, 0.005), vec2(0.0), vec2(0.22, -0.4))-0.07;
  float d5 = p.y - 0.205;
  float d6 = box(p - vec2(-0.155, -0.075), vec2(0.065, 0.30));
  float d7 = box(p - vec2(-0.055, 0.225), vec2(0.06, 0.1));

  float d = d1;
  d = min(d, d7);
  d = max(d, -d5);
  d = min(d, d2);
  d = min(d, d6);
  d = min(d, d3);
  d = pmax(d, -d0, 0.0125);
  p -= vec2(0.25, 0.0)+off;

  return d;
}

float atari_t(inout vec2 p, vec2 off) {
  p -= vec2(0.195, 0.0);

  float d0 = box(p - vec2(0.0, 0.265), vec2(0.195, 0.06)-0.0125)-0.0125;
  float d1 = box(p - vec2(0.0, -0.03), vec2(0.07, 0.295)-0.0125)-0.0125;

  float d = d0;
  d = pmin(d, d1, 0.0125);

  p -= vec2(0.195, 0.0) + off;

  return d;
}

float atari(vec2 p) {
  p -= vec2(-0.33, 0.0);
  float d = 1E6;
  vec2 rp = p;
  rp.x = abs(rp.x);
  rp.x -= -0.195;
  d = min(d, atari_t(rp, vec2(-0.055, 0.0)));
  d = min(d, atari_a(rp, vec2(-0.055, 0.0)));
  p.x -= 0.72;
  d = min(d, atari_r(p, vec2(0.02, 0.0)));
  d = min(d, atari_i(p, vec2(0.0, 0.0)));
  return d;
}
