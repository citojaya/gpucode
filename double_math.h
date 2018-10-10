#ifndef DOUBLE_MATH_H
#define DOUBLE_MATH_H


#include <math.h>
#include "cuda_runtime.h"




//*--- OPERATOR ---*/
// add
inline __host__ __device__ int3 operator+(int3 a, int3 b)
{
	return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}





// negate
inline __host__ __device__ double3 operator-(double3 &a)
{
	return make_double3(-a.x, -a.y, -a.z);
}

inline __host__ __device__ double4 operator-(double4 &a)
{
	return make_double4(-a.x, -a.y, -a.z, -a.w);
}



// addition

inline __host__ __device__ double3 operator+(double3 a, double3 b)
{
	return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(double3 &a, double3 b)
{
	a.x += b.x; a.y += b.y; a.z += b.z;
}
inline __host__ __device__ double3 operator+(double3 a, double b)
{
	return make_double3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ void operator+=(double3 &a, double b)
{
	a.x += b; a.y += b; a.z += b;
}



inline __host__ __device__ double3 operator+(double b, double3 a)
{
	return make_double3(a.x + b, a.y + b, a.z + b);
}

inline __host__ __device__ double4 operator+(double4 a, double4 b)
{
	return make_double4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
inline __host__ __device__ void operator+=(double4 &a, double4 b)
{
	a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}
inline __host__ __device__ double4 operator+(double4 a, double b)
{
	return make_double4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ double4 operator+(double b, double4 a)
{
	return make_double4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ void operator+=(double4 &a, double b)
{
	a.x += b; a.y += b; a.z += b; a.w += b;
}




// subtract
inline __host__ __device__ double3 operator-(double3 a, double3 b)
{
	return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ void operator-=(double3 &a, double3 b)
{
	a.x -= b.x; a.y -= b.y; a.z -= b.z;
}
inline __host__ __device__ double3 operator-(double3 a, double b)
{
	return make_double3(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ double3 operator-(double b, double3 a)
{
	return make_double3(b - a.x, b - a.y, b - a.z);
}
inline __host__ __device__ void operator-=(double3 &a, double b)
{
	a.x -= b; a.y -= b; a.z -= b;
}


inline __host__ __device__ double4 operator-(double4 a, double4 b)
{
	return make_double4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}
inline __host__ __device__ void operator-=(double4 &a, double4 b)
{
	a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
}
inline __host__ __device__ double4 operator-(double4 a, double b)
{
	return make_double4(a.x - b, a.y - b, a.z - b,  a.w - b);
}
inline __host__ __device__ void operator-=(double4 &a, double b)
{
	a.x -= b; a.y -= b; a.z -= b; a.w -= b;
}






// multiply


inline __host__ __device__ double3 operator*(double3 a, double3 b)
{
	return make_double3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ void operator*=(double3 &a, double3 b)
{
	a.x *= b.x; a.y *= b.y; a.z *= b.z;
}
inline __host__ __device__ double3 operator*(double3 a, double b)
{
	return make_double3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ double3 operator*(double b, double3 a)
{
	return make_double3(b * a.x, b * a.y, b * a.z);
}
inline __host__ __device__ void operator*=(double3 &a, double b)
{
	a.x *= b; a.y *= b; a.z *= b;
}

inline __host__ __device__ double4 operator*(double4 a, double4 b)
{
	return make_double4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}
inline __host__ __device__ void operator*=(double4 &a, double4 b)
{
	a.x *= b.x; a.y *= b.y; a.z *= b.z; a.w *= b.w;
}
inline __host__ __device__ double4 operator*(double4 a, double b)
{
	return make_double4(a.x * b, a.y * b, a.z * b,  a.w * b);
}
inline __host__ __device__ double4 operator*(double b, double4 a)
{
	return make_double4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline __host__ __device__ void operator*=(double4 &a, double b)
{
	a.x *= b; a.y *= b; a.z *= b; a.w *= b;
}




// divide

inline __host__ __device__ double3 operator/(double3 a, double3 b)
{
	return make_double3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline __host__ __device__ void operator/=(double3 &a, double3 b)
{
	a.x /= b.x; a.y /= b.y; a.z /= b.z;
}
inline __host__ __device__ double3 operator/(double3 a, double b)
{
	return make_double3(a.x / b, a.y / b, a.z / b);
}
inline __host__ __device__ void operator/=(double3 &a, double b)
{
	a.x /= b; a.y /= b; a.z /= b;
}
inline __host__ __device__ double3 operator/(double b, double3 a)
{
	return make_double3(b / a.x, b / a.y, b / a.z);
}

inline __host__ __device__ double4 operator/(double4 a, double4 b)
{
	return make_double4(a.x / b.x, a.y / b.y, a.z / b.z,  a.w / b.w);
}
inline __host__ __device__ void operator/=(double4 &a, double4 b)
{
	a.x /= b.x; a.y /= b.y; a.z /= b.z; a.w /= b.w;
}
inline __host__ __device__ double4 operator/(double4 a, double b)
{
	return make_double4(a.x / b, a.y / b, a.z / b,  a.w / b);
}
inline __host__ __device__ void operator/=(double4 &a, double b)
{
	a.x /= b; a.y /= b; a.z /= b; a.w /= b;
}
inline __host__ __device__ double4 operator/(double b, double4 a){
	return make_double4(b / a.x, b / a.y, b / a.z, b / a.w);
}




////////////////////////////////////////////////////////////////////////////////


// dot product
inline __host__ __device__ double dot(double3 a, double3 b)
{ 
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ double dot(double4 a, double4 b)
{ 
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}




// length

inline __host__ __device__ double length(double3 v)
{
	return sqrt(dot(v, v));
}
inline __host__ __device__ double length(double4 v)
{
	return sqrt(dot(v, v));
}



// cross product

inline __host__ __device__ double3 cross(double3 a, double3 b)
{ 
	return make_double3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x); 
}


inline __host__ __device__ double maxd(double a, double b)
{
	return a > b ? a : b;
}


inline __host__ __device__ double mind(double a, double b)
{
	return a < b ? a : b;
}


#endif