// The MIT License
// Copyright © 2021 Pascal Gilcher
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// Source found at: https://www.shadertoy.com/view/flSXRV

// New (?) method of approximating/remaking atan2. See main shader
// for benchmark results. Own benchmarks appreciated!

// Current methods reduce the number of newton raphson iterations
// or find new ways to reduce complexity by leveraging symmetry

// However, I seem to have found a different approach:
//
//
//         cos(atan2(y,x)) = x * rsqrt(x^2 + y^2)
//
//         therefore
//
//                     /  y > 0        acos(x * rsqrt(x^2 + y^2))
//         atan2(y,x) =|
//                     \  else        -acos(x * rsqrt(x^2 + y^2))
//
//
// This method is significantly faster than intrinsic atan2, matches
// the existing approximations in speed, yet is over 20x more precise.
//
// Additionally, I found that using 2 approximations for both rsqrt and acos
// whose errors roughly cancel each other out also yield acceptable results.
// Errors are larger than of the existing approximations, but this
// variant is over twice as fast as the existing approximations.
//
//
//        rsqrt(x^2 + y^2)   -->   rcp(abs(x) + abs(y))
//
//        acos(x)            -->   PI/2 - x * PI/2
//
// Finally, I found a variant that has comparable error to original, but only
// for a normalized xy pair. Since this might be a use case, I added it
// as well.

#define PI    3.141592654
#define PI_2  (0.5*PI)
#define PI_4  (0.25*PI)

float rcp(float x) {
  return 1.0 / x;
}

//P. Gilcher '21, accurate implementation of the concept
float newfastatan2(float y, float x) {
  vec2 tv = vec2(x, y);
  float cosatan2 = x * inversesqrt(dot(tv,tv));
  float t = acos(cosatan2);
  return y < 0.0 ? -t : t;
}

//P. Gilcher '21, strange approximation
float newfastatan2_A(float y, float x) {
  float cosatan2 = x * rcp(abs(x) + abs(y));
  float t = PI_2 - cosatan2 * PI_2;
  return y < 0.0 ? -t : t;
}

//P. Gilcher '21, better strange approximation for normalized xy
float newfastatan2_B(float y, float x) {
  float cosatan2 = (x * (0.5 * abs(x) + 0.5)) * rcp(abs(x) + abs(y) * 0.7071);
  float t = PI_2 - cosatan2 * PI_2;
  return y < 0.0 ? -t : t;
}

// Efficient approximations for the arctangent function,
// Rajan, S. Sichun Wang Inkol, R. Joyal, A., May 2006

// best code structure I found

float canonicfastatan2_A(float y, float x) {
  bool a = abs(y) < abs(x);
  float i = (a) ? (y * rcp(x)) : (x * rcp(y));
  i = i * (1.0584 + abs(i) * -0.273);
  float piadd = y > 0.0 ? PI : -PI;
  i = a ? (x < 0.0 ? piadd : 0.0) + i : 0.5 * piadd - i;
  return i;
}

//from same paper, better approximation in the middle
float canonicfastatan2_B(float y, float x) {
  bool a = abs(y) < abs(x);
  float i = (a) ? (y * rcp(x)) : (x * rcp(y));
  i = PI_4*i - i*(abs(i) - 1.0)*(0.2447 + 0.0663*abs(i));
  float piadd = y > 0.0 ? PI : -PI;
  i = a ? (x < 0.0 ? piadd : 0.0) + i : 0.5 * piadd - i;
  return i;
}