#version 300 es
// -----------------------------------------------------------------------------
// PRELUDE
// -----------------------------------------------------------------------------
precision highp float;
uniform float time;
uniform vec2 resolution;
in vec2 v_texcoord;
out vec4 fragColor;

// -----------------------------------------------------------------------------

#define TIME        time
#define TTIME       (TAU*TIME)
#define RESOLUTION  resolution

float rayTorus(vec3 ro, vec3 rd, vec2 tor) {
  float po = 1.0;
  float Ra2 = tor.x*tor.x;
  float ra2 = tor.y*tor.y;
  float m = dot(ro,ro);
  float n = dot(ro,rd);
  float k = (m + Ra2 - ra2)/2.0;
  float k3 = n;
  float k2 = n*n - Ra2*dot(rd.xy,rd.xy) + k;
  float k1 = n*k - Ra2*dot(rd.xy,ro.xy);
  float k0 = k*k - Ra2*dot(ro.xy,ro.xy);

  if(abs(k3*(k3*k3-k2)+k1) < 0.01) {
    po = -1.0;
    float tmp=k1; k1=k3; k3=tmp;
    k0 = 1.0/k0;
    k1 = k1*k0;
    k2 = k2*k0;
    k3 = k3*k0;
  }

  float c2 = k2*2.0 - 3.0*k3*k3;
  float c1 = k3*(k3*k3-k2)+k1;
  float c0 = k3*(k3*(c2+2.0*k2)-8.0*k1)+4.0*k0;
  c2 /= 3.0;
  c1 *= 2.0;
  c0 /= 3.0;
  float Q = c2*c2 + c0;
  float R = c2*c2*c2 - 3.0*c2*c0 + c1*c1;
  float h = R*R - Q*Q*Q;

  if(h>=0.0) {
    h = sqrt(h);
    float v = sign(R+h)*pow(abs(R+h),1.0/3.0); // cube root
    float u = sign(R-h)*pow(abs(R-h),1.0/3.0); // cube root
    vec2 s = vec2((v+u)+4.0*c2, (v-u)*sqrt(3.0));
    float y = sqrt(0.5*(length(s)+s.x));
    float x = 0.5*s.y/y;
    float r = 2.0*c1/(x*x+y*y);
    float t1 =  x - r - k3; t1 = (po<0.0)?2.0/t1:t1;
    float t2 = -x - r - k3; t2 = (po<0.0)?2.0/t2:t2;
    float t = 1e20;
    if(t1>0.0) t=t1;
    if(t2>0.0) t=min(t,t2);
    return t;
  }

  float sQ = sqrt(Q);
  float w = sQ*cos(acos(-R/(sQ*Q)) / 3.0);
  float d2 = -(w+c2); if(d2<0.0) return -1.0;
  float d1 = sqrt(d2);
  float h1 = sqrt(w - 2.0*c2 + c1/d1);
  float h2 = sqrt(w - 2.0*c2 - c1/d1);
  float t1 = -d1 - h1 - k3; t1 = (po<0.0)?2.0/t1:t1;
  float t2 = -d1 + h1 - k3; t2 = (po<0.0)?2.0/t2:t2;
  float t3 =  d1 - h2 - k3; t3 = (po<0.0)?2.0/t3:t3;
  float t4 =  d1 + h2 - k3; t4 = (po<0.0)?2.0/t4:t4;
  float t = 1e20;
  if(t1>0.0) t=t1;
  if(t2>0.0) t=min(t,t2);
  if(t3>0.0) t=min(t,t3);
  if(t4>0.0) t=min(t,t4);
  return t;
}

vec3 torusNormal(vec3 pos, vec2 tor) {
  return normalize( pos*(dot(pos,pos)-tor.y*tor.y - tor.x*tor.x*vec3(1.0,1.0,-1.0)));
}


vec3 color(vec2 p, vec2 q) {
  const float rdd = 2.0;
  vec3 ro = 2.0*vec3(0.0, 0.8, 1.0);
  vec3 la = vec3(0.0, 0.0, 0.0);
  vec3 up = vec3(0.0, 0.0, 1.0);

  vec3 ww = normalize(la - ro);
  vec3 uu = normalize(cross(up, ww));
  vec3 vv = normalize(cross(ww,uu));
  vec3 rd = normalize(p.x*uu + p.y*vv + rdd*ww);

  const vec2 tor = 0.5*vec2(1.0, 0.5);
  float td    = rayTorus(ro, rd, tor);
  vec3  tpos  = ro + rd*td;
  vec3  tnor  = torusNormal(tpos, tor);

  vec3 col = vec3(0.0);
  if (td > -1.0) {
    col += abs(tnor);
  }


  return col;
}

vec3 postProcess(vec3 col, vec2 q) {
  col = clamp(col, 0.0, 1.0);
  col = pow(col, 1.0/vec3(2.2));
  col = col*0.6+0.4*col*col*(3.0-2.0*col);
  col = mix(col, vec3(dot(col, vec3(0.33))), -0.4);
  col *=0.5+0.5*pow(19.0*q.x*q.y*(1.0-q.x)*(1.0-q.y),0.7);
  return col;
}

void main(void) {
  vec2 q = v_texcoord;
  vec2 p = -1. + 2. * q;
  p.x *= RESOLUTION.x/RESOLUTION.y;
  vec3 col = color(p, q);
  col = postProcess(col, q);
  fragColor = vec4(col, 1.0);
}
