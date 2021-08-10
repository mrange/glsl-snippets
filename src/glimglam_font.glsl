// -----------------------------------------------------------------------------
// Licenses referenced:
//  CC0   - https://creativecommons.org/share-your-work/public-domain/cc0/
//  MIT   - https://mit-license.org/
//  WTFPL - https://en.wikipedia.org/wiki/WTFPL
// -----------------------------------------------------------------------------

// Glimglam distance field font

const float glimglam_corner0 = 0.02;
const float glimglam_corner1 = 0.075;
const float glimglam_topy    = 0.0475+glimglam_corner0*0.5;
const float glimglam_smoother= 0.0125;

// License: MIT, author: Inigo Quilez, found: https://www.iquilezles.org/www/articles/smin/smin.htm
float pmin(float a, float b, float k) {
  float h = clamp(0.5+0.5*(b-a)/k, 0.0, 1.0);
  return mix(b, a, h) - k*h*(1.0-h);
}

// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
float pmax(float a, float b, float k) {
  return -pmin(-a, -b, k);
}

float glimglam_bar(vec2 p) {
  vec2 pbar = p;
  pbar.y -= topy;
  return abs(pbar.y)-corner0;
}

float glimglam_a(vec2 p) {
  p.x = abs(p.x);
  float db = roundedBox(p, vec2 (0.19, 0.166), vec4(corner1, corner0, corner1, corner0));
  float dc = corner(p-vec2(0.045, -0.07))-corner0;

  float d = db;
  d = max(d, -dc);

  return d;
}

float glimglam_c(vec2 p) {
  p = -p.yx;
  float db = roundedBox(p, vec2 (0.166, 0.19), vec4(corner1, corner0, corner1, corner0));
  p.x = abs(p.x);
  float dc = corner(p-vec2(0.055, topy))-corner0;

  float d = db;
  d = max(d, -dc);

  return d;
}

float glimglam_e(vec2 p) {
  p = -p.yx;
  float db = roundedBox(p, vec2 (0.166, 0.19), vec4(corner0, corner0, corner0, corner0));

  float dl = abs(p.x-(0.075-corner0))-corner0;
  float dt = p.y-topy;

  float d = db;
  d = max(d, -pmax(dl,dt, smoother));

  return d;
}

float glimglam_g(vec2 p) {
  float db = roundedBox(p, vec2 (0.19, 0.166), vec4(corner0, corner1, corner1, corner1));
  float dc = corner(-(p-vec2(-0.045, -0.055)));
  dc = abs(dc) - corner0;
  float dd = max(p.x-0.065, p.y-topy);
  float d = db;
  d = max(d, -max(dc, dd));
  return d;
}

float glimglam_h(vec2 p) {
  p.x = abs(p.x);
  float da = roundedBox(p-vec2(0.13, 0.0), vec2 (0.066, 0.166), vec4(corner0));
  float db = roundedBox(p, vec2 (0.16, 0.05), vec4(corner0));
  float d = da;
  d = min(d, db);
  return d;
}

float glimglam_i(vec2 p) {
  return roundedBox(p, vec2 (0.066, 0.166), vec4(corner0));
}

float glimglam_j(vec2 p) {
  p.x = -p.x;
  float db = roundedBox(p, vec2 (0.15, 0.166), vec4(corner0, corner0, corner0, corner1));
  float dc = corner(-(p-vec2(-0.007, -0.055)))-corner0;
  float d = db;
  d = max(d, -dc);
  return d;
}

float glimglam_l(vec2 p) {
  float db = roundedBox(p, vec2 (0.175, 0.166), vec4(corner0, corner0, corner0, corner1));
  float dc = corner(-(p-vec2(-0.027, -0.055)))-corner0;
  float d = db;
  d = max(d, -dc);
  return d;
}

float glimglam_m(vec2 p) {
  float db = roundedBox(p, vec2 (0.255, 0.166), vec4(corner1, corner0, corner0, corner0));
  p.x = abs(p.x);
  float dl = abs(p.x-0.095)-corner0*2.0;
  float dt = p.y-topy;

  float d = db;
  d = max(d, -max(dl,dt));

  return d;
}

float glimglam_n(vec2 p) {
  float db = roundedBox(p, vec2 (0.19, 0.166), vec4(corner1, corner0, corner0, corner0));

  float dl = abs(p.x)-0.07;
  float dt = p.y-topy;

  float d = db;
  d = max(d, -max(dl,dt));

  return d;
}

float glimglam_o(vec2 p) {
  const float sz = 0.05;
  float db = roundedBox(p, vec2(0.19, 0.166)-sz, vec4(corner1, corner1, corner1, corner1)-sz);
  db = abs(db)-sz;

  float d = db;

  return d;
}

float glimglam_s(vec2 p) {
  p.x = -p.x;
  p = -p.yx;
  float db = roundedBox(p, vec2 (0.166, 0.19), vec4(corner1, corner0, corner0, corner1));
  vec2 pc = p;
  pc.x *= sign(pc.y);
  pc.y = abs(pc.y);
  float cr = corner1*1.3;
  pc -=vec2(-0.055, 0.20);
  pc.x = -pc.x;
  float dc = corner(pc+cr)-cr;
  vec2 pk = p;
  pk = -abs(pk);
  float dk = pk.x+topy;
  dc = min(dk, dc);

  float dl = abs(p.x-(0.075-corner0))-corner0;
  float dt = p.y-topy;

  float d = db;
  d = max(d, -pmax(dl,dt, smoother));
  d = pmax(d, dc, smoother);

  return d;
}

float glimglam_t(vec2 p) {
  float da = roundedBox(p-vec2(0.0, 0.12), vec2 (0.166, 0.05), vec4(corner0));
  float db = roundedBox(p, vec2 (0.066, 0.166), vec4(corner0));
  float d = da;
  d = min(d, db);
  return d;
}

float glimglam_z(vec2 p) {
  p = -p.yx;
  float db = roundedBox(p, vec2 (0.166, 0.19), vec4(corner0, corner0, corner0, corner0));
  vec2 pc = p;
  pc.x *= sign(pc.y);
  pc.y = abs(pc.y);
  float cr = corner1*1.3;
  pc -=vec2(-0.055, 0.20);
  pc.x = -pc.x;
  float dc = corner(pc+cr)-cr;
  vec2 pk = p;
  pk = -abs(pk);
  float dk = pk.x+topy;
  dc = min(dk, dc);

  float dl = abs(p.x-(0.075-corner0))-corner0;
  float dt = p.y-topy;

  float d = db;
  d = max(d, -pmax(dl,dt, smoother));
  d = pmax(d, dc, smoother);

  return d;
}

float glimglam(vec2 p) {
  float dbar = glimglam_bar(p);

  vec2 pg = p;
  pg.x -= -0.665;
  pg.x = -abs(pg.x);
  pg.x -= -0.7475;
  pg.x *= -sign(p.x+0.665);
  float dg = glimglam_g(pg);

  vec2 pi = p;
  pi.x -= -0.746;
  float di = glimglam_i(pi);

  vec2 pl = p;
  pl.x -= -0.27;
  pl.x = -abs(pl.x);
  pl.x -= -0.745;
  pl.x *= -sign(p.x+0.27);
  float dl = glimglam_l(pl);

  vec2 pa = p;
  pa.x -= 0.87;
  float da = glimglam_a(pa);

  vec2 pm = p;
  pm.x -= 0.475;
  pm.x = abs(pm.x);
  pm.x -= 0.875;
  pm.x *= sign(p.x-0.475);
  float dm = glimglam_m(pm);

  float d = 1E6;
  d = min(d, dg);
  d = min(d, dl);
  d = min(d, di);
  d = min(d, da);
  d = min(d, dm);
  d = pmax(d, -dbar, smoother);

  return d;
}

float lance(vec2 p) {
  p.x -= -0.810;
  float dbar = glimglam_bar(p);

  vec2 pl = p;
  float dl = glimglam_l(pl);

  vec2 pa = p;
  pa.x -= 0.39;
  float da = glimglam_a(pa);

  vec2 pn = p;
  pn.x -= 0.795;
  float dn = glimglam_n(pn);

  vec2 pc = p;
  pc.x -= 1.2;
  float dc = glimglam_c(pc);

  vec2 pe = p;
  pe.x -= 1.605;
  float de = glimglam_e(pe);

  float d = 1E6;
  d = min(d, dl);
  d = min(d, da);
  d = min(d, dn);
  d = min(d, dc);
  d = min(d, de);
  d = pmax(d, -dbar, smoother);

  return d;
}

float jez(vec2 p) {
  p.x -= -0.401;
  float dbar = glimglam_bar(p);

  vec2 pj = p;
  float dj = glimglam_j(pj);

  vec2 pe = p;
  pe.x -= 0.36;
  float de = glimglam_e(pe);

  vec2 pz = p;
  pz.x -= 0.76;
  float dz = glimglam_z(pz);

  float d = 1E6;
  d = min(d, dj);
  d = min(d, de);
  d = min(d, dz);
  d = pmax(d, -dbar, smoother);
  return d;
}

float longshot(vec2 p) {
  p.x -= -1.385;
  float dbar = glimglam_bar(p);

  vec2 pl = p;
  float dl = glimglam_l(pl);

  vec2 po = p;
  po -= vec2(1.395, 0.0);
  po.x = abs(po.x);
  po -= vec2(1.0125, 0.0);
  float do_ = glimglam_o(po);

  vec2 pn = p;
  pn -= vec2(0.785, 0.0);
  float dn = glimglam_n(pn);

  vec2 pg = p;
  pg -= vec2(1.185, 0.0);
  float dg = glimglam_g(pg);

  vec2 ps = p;
  ps -= vec2(1.585, 0.0);
  float ds = glimglam_s(ps);

  vec2 ph = p;
  ph -= vec2(1.995, 0.0);
  float dh = glimglam_h(ph);

  vec2 pt = p;
  pt -= vec2(2.78, 0.0);
  float dt = glimglam_t(pt);

  float d = 1E6;
  d = min(d, dl);
  d = min(d, do_);
  d = min(d, dn);
  d = min(d, dg);
  d = min(d, ds);
  d = min(d, dh);
  d = min(d, dt);
  d = pmax(d, -dbar, smoother);
  return d;
}


