// Found: https://fragcoord.xyz/s/pxmcvnpc
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 @lumiey
//[LICENSE] https://opensource.org/licenses/MIT

/////////// HASHES ///////////
// Fi Hash
uniform sampler2D u_tex1; //https://cdn.jsdelivr.net/gh/otaviogood/shader_fontgen@master/codepage12.png
float hash11(float p) {
    uint u = floatBitsToUint(p * 3141592653.0);
    return float(u * u * 3141592653u) / float(~0u);
}

float hash12(vec2 p) {
    uvec2 u = floatBitsToUint(p * vec2(141421356, 2718281828));
    return float((u.x ^ u.y) * 3141592653u) / float(~0u);
}

vec2 hash22(vec2 p) {
    uvec2 u = floatBitsToUint(p * vec2(141421356, 2718281828));
    return vec2((u.x ^ u.y) * uvec2(3141592653, 1618033988)) / float(~0u);
}

vec3 hash32(vec2 p) {
    uvec2 u = floatBitsToUint(p * vec2(141421356, 2718281828));
    return vec3((u.x ^ u.y) * uvec3(1732050807, 2645751311, 3316624790)) / float(~0u);
}

float hash13(vec3 p) {
    uvec3 u = floatBitsToUint(p * vec3(141421356, 2718281828, 1618033988));
    return float((u.x ^ u.y ^ u.z) * 3141592653u) / float(~0u);
}

vec3 hash33(vec3 p) {
    uvec3 u = floatBitsToUint(p * vec3(141421356, 2718281828, 1618033988));
    return vec3((u.x ^ u.y ^ u.z) * uvec3(1732050807, 2645751311, 3316624790)) / float(~0u);
}

/////////// 1D NOISE ///////////
float value11(float p) {
	float i = floor(p);
	return mix(hash11(i), hash11(i + 1.0), p - i);
}

/////////// 2D NOISE ///////////
float value12(vec2 p) {
	vec2 i = floor(p);
	vec2 f = p - i;
	f *= f * (3.0 - 2.0 * f);
	float res = mix(
		mix(hash12(i), hash12(i + vec2(1, 0)), f.x),
		mix(hash12(i + vec2(0, 1)), hash12(i + vec2(1)), f.x), f.y);
	return res;
}

float perlin12(vec2 p) {
    vec2 i = floor(p);
    vec2 f = p - i;
    vec2 u = f * f * f * (10.0 + f * (6.0 * f - 15.0));
    float a = dot(normalize(hash22(i + vec2(0, 0)) - 0.5), f - vec2(0, 0));
    float b = dot(normalize(hash22(i + vec2(1, 0)) - 0.5), f - vec2(1, 0));
    float c = dot(normalize(hash22(i + vec2(0, 1)) - 0.5), f - vec2(0, 1));
    float d = dot(normalize(hash22(i + vec2(1, 1)) - 0.5), f - vec2(1, 1));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y) * 0.7 + 0.5;
}

vec3 perlin12d(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    vec2 du = 30.0 * f * f * (f * (f - 2.0) + 1.0);
    vec2 ga = hash22(i + vec2(0, 0)) * 2.0 - 1.0;
    vec2 gb = hash22(i + vec2(1, 0)) * 2.0 - 1.0;
    vec2 gc = hash22(i + vec2(0, 1)) * 2.0 - 1.0;
    vec2 gd = hash22(i + vec2(1, 1)) * 2.0 - 1.0;
    float va = dot(ga, f - vec2(0, 0));
    float vb = dot(gb, f - vec2(1, 0));
    float vc = dot(gc, f - vec2(0, 1));
    float vd = dot(gd, f - vec2(1, 1));
    return vec3(va + u.x * (vb - va) + u.y * (vc - va) + u.x * u.y * (va - vb - vc + vd), ga + u.x * (gb - ga) + u.y * (gc - ga) + u.x * u.y * (ga - gb - gc + gd) + du * (u.yx * (va - vb - vc + vd) + vec2(vb, vc) - va));
}

float simplex12(vec2 p) {
	vec2 i = floor(p + (p.x + p.y) * 0.366025);
    vec2 a = p - i + (i.x + i.y) * 0.211324;
    float m = step(a.y, a.x);
    vec2 o = vec2(m, 1.0 - m);
    vec2 b = a - o + 0.211324;
	vec2 c = a - 0.577351;
    vec3 h = max(0.5 - vec3(dot(a, a), dot(b, b), dot(c, c)), 0.0);
	vec3 n = h * h * h * h *
        vec3(dot(a, hash22(i) - 0.5),
             dot(b, hash22(i + o) - 0.5),
             dot(c, hash22(i + 1.0) - 0.5));
    return dot(n, vec3(70)) + 0.5;
}

float worley12(vec2 p) {
    vec2 i = floor(p);
    p -= i;
    float w = 1e9;
    for (float x = -1.0; x <= 1.0; ++x)
    for (float y = -1.0; y <= 1.0; ++y) {
        vec2 c = p - vec2(x, y) - hash12(i + vec2(x, y));
       	w = min(w, dot(c, c));
    }
    return 1.0 - sqrt(w);
}

// s: edge smoothness
float voronoi12(vec2 x, float s) {
    s = 1.0 / s;
    vec2 p = floor(x);
    vec2 f = x - p;
	float va = 0.0;
	float wt = 0.0;
    for(float x = -1.0; x <= 1.0; x++)
    for(float y = -1.0; y <= 1.0; y++) {
		vec3 o = hash32(p + vec2(x, y));
		float d = length(vec2(x, y) - f + o.xy);
		float ww = pow(smoothstep(1.414, 0.0, d), s);
		va += o.z * ww;
		wt += ww;
    }
    return va / wt;
}

float blue12(vec2 p) {
    float v = 0.0;
    for (int k = 0; k < 9; k++)
        v += hash12(p + vec2(k % 3 - 1, k / 3 - 1));
    return 0.9 * (1.125 * hash12(p) - v / 8.0) + 0.5;
}

int hilbert_encode(int n, ivec2 p) {
    int i = 0;
    for (int s = n >> 1; s > 0; s >>= 1) {
        int rx = int((p.x & s) != 0);
        int ry = int((p.y & s) != 0);
        i += s * s * ((rx << 1) | rx ^ ry);
        p ^= (p.x ^ p.y) * (1 - ry) ^ (s - 1) * (rx & (1 - ry));
    }
    return i;
}

float hilbert_blue12(vec2 p) {
    return fract(0.6180339887498948482 * float(hilbert_encode(512, ivec2(p)) % 262144));
}

// Modified from: https://www.shadertoy.com/view/XsGBDt
float crater12(vec2 p) {
    vec2 f = fract(p);
    p = floor(p);
    float va = 0.;
    float wt = 0.;
    for (int i = -2; i <= 2; i++)
        for (int j = -2; j <= 2; j++) {
                vec2 g = vec2(i, j);
                vec2 o = hash22(p + g);
                float d = distance(f - g, o);
                float w = exp(-4. * d);
                va += w * sin(6.28 * sqrt(max(d, 0.06)));
                wt += w;
            }
    return abs(va / wt);
}

float gabor12(vec2 p) {
    const float kF = 8.0;
    vec2 i = floor(p);
	vec2 f = p - i;
    f *= f * (3.0 - 2.0 * f);
    return mix(mix(sin(kF * dot(p, hash22(i + vec2(0, 0)))),
               	   sin(kF * dot(p, hash22(i + vec2(1, 0)))), f.x),
               mix(sin(kF * dot(p, hash22(i + vec2(0, 1)))),
               	   sin(kF * dot(p, hash22(i + vec2(1, 1)))), f.x), f.y);
}

vec2 curl22(vec2 p) {
    vec2 e = vec2(0.1, 0);
    vec2 a = vec2(perlin12(p + e.xy), perlin12(p + e.yx));
    vec2 b = vec2(perlin12(p - e.xy), perlin12(p - e.yx));
    return (a - b) / e.x * 0.5;
}

// Inspired from: https://www.shadertoy.com/view/4syXRD
float scratch(vec2 uv, float f) {
    vec2 seed = floor(uv);
    uv -= seed;
    seed.x = floor(sin(seed.x * 51024.0) * 3104.0);
    seed.y = floor(sin(seed.y * 1324.0) * 554.0);

    uv = uv * 2.0 - 1.0;
    uv = uv * cos(seed.x + seed.y) + vec2(-uv.y, uv.x) * sin(seed.x + seed.y);
    uv += sin(seed.x - seed.y);
    uv = uv * 0.5 + 0.5;

    const float WAVYNESS = 0.2;
    float s = (sin(seed.x + uv.y * 3.1415) + sin(seed.y + uv.y * 3.1415)) * WAVYNESS;

    float x = abs(uv.x - 0.5 + s);
    x = 0.5 - x * f;
    x = smoothstep(-2.0, fwidth(x) * 1.5 + 16.0, x) * 12.0;
    x *= uv.y;

    return x;
}

float scratches12(vec2 uv) {
    float scratches = 0.0;
    float f = 1.0 / length(fwidth(uv));
    for(int i = 0; i < 8; ++i) {
        float x = scratch(uv, f);
    	scratches = max(scratches, x);
        uv = uv * mat2(1.0, 0.7, -0.7, 1.0) - 12.31;
    }
    return scratches;
}

// https://www.shadertoy.com/view/wsBfzK
float wavelet12(vec2 p, float phase, float scale) {
    float d = 0.0, s = 1.0, m = 0.0, a;
    for (float i = 0.0; i < 4.0; ++i) {
        vec2 q = p * s, g = fract(floor(q) * vec2(123.34, 233.53));
    	g += dot(g, g + 23.234);
		a = fract(g.x * g.y) * 1e3;// +z*(mod(g.x+g.y, 2.)-1.); // add vorticity
        q = (fract(q) - 0.5) * mat2(cos(a), -sin(a), sin(a), cos(a));
        d += sin(q.x * 10.0 + phase) * smoothstep(0.25, 0.0, dot(q, q)) / s;
        p = p * mat2(0.54, -0.84, 0.84, 0.54) + i;
        m += 1.0 / s;
        s *= scale;
    }
    return d / m;
}

vec3 gullies(vec2 p, vec2 slope) {
    vec2 side_dir = vec2(-slope.y, slope.x) * 3.14159265;
    vec2 id = floor(p);
    p -= id;
    vec2 height_slope = vec2(0);
    float w_sum = 0.0;
    for(int x = -1; x <= 2; x++) {
        for(int y = -1; y <= 2; y++) {
            vec2 off = vec2(x, y);
            vec2 c = p - off - hash22(id + off) + 0.5;
            float dist2 = dot(c, c);
            float w = max(0.0, exp(-dist2 * 2.0) - 0.01111);
            w_sum += w;
            float t = dot(c, side_dir);
            height_slope += vec2(cos(t), -sin(t)) * w;
        }
    }
    return vec3(height_slope.x, height_slope.y * side_dir) / w_sum;
}

// modified & simplified from: https://www.shadertoy.com/view/sf23W1
vec3 erosion12(vec2 p) {
    vec3 nd = perlin12d(p);
    float strength = 0.25, freq = 8.0, total = 1.0;
    for(int i = 0; i < 4; i++) {
        float len2 = dot(nd.yz, nd.yz);
        nd += gullies(p * freq, nd.yz * pow(len2, 0.5 * (0.5 - 1.0))) * strength * vec3(1, freq, freq);
        total += strength;
        strength *= 0.5;
        freq *= 2.0;
    }
    return nd / total;
}

vec2 fbm_paper(vec2 p, int octaves) {
	vec2 s = vec2(0);
    float m = 0.0, a = 1.0;
	for(int i = 0; i < octaves; i++) {
		s += a * clamp(curl22(p) * 0.5 + 0.5, vec2(0),  vec2(1));
		m += a;
        a *= 0.8;
        p *= 2.0;
	}
	return s / m;
}

float paper12(vec2 p) {
    return length(fbm_paper(p, 10)) / 1.414 * 0.6 + 0.4;
}

float fbm12(vec2 p, int octaves) {
    float s = 0.0, m = 0.0, a = 1.0;
	for (int i = 0; i < octaves; i++) {
        float n = perlin12(p);
		s += a * n;
		m += a;
		a *= 0.5;
		p *= 2.0;
	}
	return s / m;
}

vec3 fbm12d(vec2 p, int octaves) {
    vec3 s = vec3(0);
    float m = 0.0, a = 1.0, f = 1.0;
	for (int i = 0; i < octaves; i++) {
        vec3 n = perlin12d(p * f);
		s += a * vec3(1, f, f) * n;
		m += a;
		a *= 0.5;
		f *= 2.0;
	}
	return s / vec3(m, 1, 1);
}

vec3 fbm_stone(vec2 p, int octaves) {
    vec3 s = vec3(0);
    float a = 1.0;
    for(int i = 0; i < 6; ++i) {
        s += a * perlin12d(p);
        a *= 0.5;
        p *= 2.0;
    }
    return s;
}

float stone12(vec2 p) {
    return fbm12(p + fbm_stone(p, 6).yz * 0.4, 6);
}

vec2 fbm_wool(vec2 p, int octaves) {
	vec2 s = vec2(0.0);
    float m = 0.0, a = 1.0;
	for(int i = 0; i < octaves; i++) {
        vec2 n = curl22(p);
		s += a * n;
		m += a;
		a *= 0.5;
		p *= 2.0;
	}
	return s / m;
}

float wool12(vec2 p) {
    vec2 n = fbm_wool(p, 6);
    return max(abs(n.x), abs(n.y));
}

/////////// 3D NOISE ///////////
vec4 perm(vec4 x) { x *= x * 34.0 + 1.0; return x - floor(x / 289.0) * 289.0; }
float value13(vec3 p) {
    vec3 a = floor(p);
    vec3 d = p - a;
    d *= d * (3.0 - 2.0 * d);
    vec4 b = a.xxyy + vec4(0, 1, 0, 1);
    vec4 k1 = perm(b.xyxy);
    vec4 k2 = perm(k1.xyxy + b.zzww) + a.zzzz;
    vec4 k3 = perm(k2);
    vec4 k4 = perm(k2 + 1.0);
    vec4 o1 = fract(k3 * 0.02439024);
    vec4 o2 = fract(k4 * 0.02439024);
    vec4 o3 = mix(o1, o2, d.z);
    vec2 o4 = mix(o3.xz, o3.yw, d.x);
    return mix(o4.x, o4.y, d.y);
}

float perlin13(vec3 p) {
    vec3 i = floor(p);
    vec3 f = p - i;
    vec3 u = f * f * f * (10.0 + f * (6.0 * f - 15.0));
    float a0 = dot(f - vec3(0, 0, 0), normalize(hash33(i + vec3(0, 0, 0)) - 0.5));
    float b0 = dot(f - vec3(1, 0, 0), normalize(hash33(i + vec3(1, 0, 0)) - 0.5));
    float c0 = dot(f - vec3(0, 1, 0), normalize(hash33(i + vec3(0, 1, 0)) - 0.5));
    float d0 = dot(f - vec3(1, 1, 0), normalize(hash33(i + vec3(1, 1, 0)) - 0.5));
    float a1 = dot(f - vec3(0, 0, 1), normalize(hash33(i + vec3(0, 0, 1)) - 0.5));
    float b1 = dot(f - vec3(1, 0, 1), normalize(hash33(i + vec3(1, 0, 1)) - 0.5));
    float c1 = dot(f - vec3(0, 1, 1), normalize(hash33(i + vec3(0, 1, 1)) - 0.5));
    float d1 = dot(f - vec3(1, 1, 1), normalize(hash33(i + vec3(1, 1, 1)) - 0.5));
    float z0 = mix(mix(a0, b0, u.x), mix(c0, d0, u.x), u.y);
    float z1 = mix(mix(a1, b1, u.x), mix(c1, d1, u.x), u.y);
    return mix(z0, z1, u.z) * 0.7 + 0.5;
}

float simplex13(vec3 p) {
	 vec3 s = floor(p + dot(p, vec3(1.0 / 3.0)));
	 vec3 x = p - s + dot(s, vec3(1.0 / 6.0));
	 vec3 e = step(vec3(0), x - x.yzx);
	 vec3 i1 = e * (1.0 - e.zxy);
	 vec3 i2 = 1.0 - e.zxy * (1.0 - e);
	 vec3 x1 = x - i1 + 1.0 / 6.0;
	 vec3 x2 = x - i2 + 1.0 / 3.0;
	 vec3 x3 = x - 0.5;
	 vec4 w = vec4(dot(x, x), dot(x1, x1), dot(x2, x2), dot(x3, x3));
	 w = max(0.6 - w, 0.0);
	 vec4 d = vec4(dot(hash33(s) - 0.5, x),
                   dot(hash33(s + i1) - 0.5, x1),
            	   dot(hash33(s + i2) - 0.5, x2),
                   dot(hash33(s + 1.0) - 0.5, x3));
	 w *= w;
	 w *= w;
	 d *= w;
	 return dot(d, vec4(26)) + 0.5;
}

float worley13(vec3 p) {
    vec3 i = floor(p);
    p -= i;
    float w = 1e9;
    for (float x = -1.0; x <= 1.0; ++x)
    for (float y = -1.0; y <= 1.0; ++y)
    for (float z = -1.0; z <= 1.0; ++z) {
        vec3 c = p - vec3(x, y, z) - hash13(i + vec3(x, y, z));
       	w = min(w, dot(c, c));
    }
    return 1.0 - sqrt(w);
}

/////////// Visualization Helpers ///////////
#define fbm12(uv, noise_fn) do {\
    vec2 p = uv;\
    float s = 0.0, m = 0.0, a = 1.0;\
	for (int i = 0; i < 6; i++) {\
        float n = noise_fn(p);\
		s += a * n;\
		m += a;\
		a *= 0.5;\
		p *= 2.0;\
	}\
	c += s / m;\
} while(false)

#define fbm12_deriv(uv, noise_fn) do {\
    vec2 p = uv;\
    vec3 s = vec3(0);\
    float m = 0.0, a = 1.0, f = 1.0;\
	for (int i = 0; i < 6; i++) {\
        vec3 n = noise_fn(p * f);\
		s += a * vec3(1, f, f) * n;\
		m += a;\
		a *= 0.25;\
		f *= 2.0;\
	}\
	c += s / vec3(m, 1, 1);\
} while(false)

float wavelet12_helper(vec2 p) {
    return wavelet12(p, u_time, 1.24) * 0.5 + 0.5;
}

float text(vec2 p, float glyph) {
	p = abs(p.x - 0.5) > 0.5 || abs(p.y - 0.5) > 0.5 ? vec2(0) : p;
	return 2.0 * (texture(u_tex1, p / 16.0 + fract(vec2(glyph, 15.0 - floor(glyph / 16.0)) / 16.0)).w - 127.0 / 255.0);
}

#define C(ch) c = mix(c, vec3(0, 0.05, 0.3), smoothstep(w, 0.0, text(f * 7.5 - vec2(x, 0), float(ch)) / 7.5 + w * 0.3)); x += 0.45;

void main() {
    float mr = min(u_resolution.x, u_resolution.y);
    vec2 uv = (gl_FragCoord.xy * 2.0 - u_resolution.xy) / mr;

    vec2 grid = vec2(6, 4);
    int id = int(gl_FragCoord.x / u_resolution.x * grid.x) + int(gl_FragCoord.y / u_resolution.y * grid.y) * int(grid.x);
    vec2 f = fract(gl_FragCoord.xy / u_resolution.xy * grid) * (u_resolution.x / grid.x) / (u_resolution.y / grid.y);
    float w = 1.0 / mr * max(grid.x, grid.y);
    vec2 p = uv * 6.0;
    float x = 0.0;
    vec3 c;
    switch (id) { // 2D noises
        case 0: c += value12(p); C(48)C(46)C(32)C(86)C(65)C(76)C(85)C(69) break;
        case 1: fbm12(p, value12); C(49)C(46)C(32)C(86)C(65)C(76)C(85)C(69)C(32)C(70)C(66)C(77) break;
        case 2: c += perlin12(p); C(50)C(46)C(32)C(80)C(69)C(82)C(76)C(73)C(78) break;
        case 3: fbm12(p, perlin12); C(51)C(46)C(32)C(80)C(69)C(82)C(76)C(73)C(78)C(32)C(70)C(66)C(77) break;
        case 4: c += simplex12(p); C(52)C(46)C(32)C(83)C(73)C(77)C(80)C(76)C(69)C(88) break;
        case 5: fbm12(p, simplex12); C(53)C(46)C(32)C(83)C(73)C(77)C(80)C(76)C(69)C(88)C(32)C(70)C(66)C(77) break;
        case 6: c += worley12(p); C(54)C(46)C(32)C(87)C(79)C(82)C(76)C(69)C(89) break;
        case 7: fbm12(p, worley12); C(55)C(46)C(32)C(87)C(79)C(82)C(76)C(69)C(89)C(32)C(70)C(66)C(77) break;
        case 8: c += blue12(floor(gl_FragCoord.xy)); C(56)C(46)C(32)C(66)C(76)C(85)C(69) break;
        case 9: c += hilbert_blue12(floor(gl_FragCoord.xy)); C(57)C(46)C(32)C(72)C(73)C(76)C(66)C(69)C(82)C(84)C(32)C(66)C(76)C(85)C(69) break;
        case 10: c += crater12(p); C(49)C(48)C(46)C(32)C(67)C(82)C(65)C(84)C(69)C(82) break;
        case 11: fbm12(p, crater12); C(49)C(49)C(46)C(32)C(67)C(82)C(65)C(84)C(69)C(82)C(32)C(70)C(66)C(77) break;
        case 12: c += gabor12(p)*.5+.5; C(49)C(50)C(46)C(32)C(71)C(65)C(66)C(79)C(82) break;
        case 13: fbm12(p, gabor12); c=c*.5+.5; C(49)C(50)C(46)C(32)C(71)C(65)C(66)C(79)C(82)C(32)C(70)C(66)C(77)  break;
        case 14: c += scratches12(p); C(49)C(52)C(46)C(32)C(83)C(67)C(82)C(65)C(84)C(67)C(72) break;
        case 15: fbm12(p, scratches12); C(49)C(52)C(46)C(32)C(83)C(67)C(82)C(65)C(84)C(67)C(72)C(32)C(70)C(66)C(77) break;
        case 16: c += wavelet12_helper(p); C(49)C(54)C(46)C(32)C(87)C(65)C(86)C(69)C(76)C(69)C(84) break;
        case 17: fbm12(p, wavelet12_helper); C(49)C(54)C(46)C(32)C(87)C(65)C(86)C(69)C(76)C(69)C(84)C(32)C(70)C(66)C(77) break;
        case 18: c += erosion12(p).x*.5+.5; C(49)C(56)C(46)C(32)C(69)C(82)C(79)C(83)C(73)C(79)C(78) break;
        case 19: c += length(curl22(p)) / 1.414; C(49)C(57)C(46)C(32)C(67)C(85)C(82)C(76) break;
        case 20: c += paper12(uv * 4.0); C(50)C(48)C(46)C(32)C(80)C(65)C(80)C(69)C(82) break;
        case 21: c += stone12(p); C(50)C(49)C(46)C(32)C(83)C(84)C(79)C(78)C(69) break;
        case 22: c += wool12(p);  C(50)C(49)C(46)C(32)C(87)C(79)C(79)C(76) break;
    }
    fragColor = vec4(c, 1);
}