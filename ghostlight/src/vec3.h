// ============================================================================
// vec3.h — minimal 3D vector type for ghostlight (CPU and GPU)
// ============================================================================
#pragma once

// Strip CUDA qualifiers when building with a regular C++ compiler so this
// header is safe to include from both .cpp and .cu translation units.
#ifndef __CUDACC__
  #ifndef __host__
    #define __host__
  #endif
  #ifndef __device__
    #define __device__
  #endif
#endif

#include <cmath>

struct Vec3f
{
    float x, y, z;

    __host__ __device__ Vec3f() : x(0), y(0), z(0) {}
    __host__ __device__ Vec3f(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    __host__ __device__ explicit Vec3f(float s) : x(s), y(s), z(s) {}

    __host__ __device__ Vec3f operator+(const Vec3f &v) const { return {x + v.x, y + v.y, z + v.z}; }
    __host__ __device__ Vec3f operator-(const Vec3f &v) const { return {x - v.x, y - v.y, z - v.z}; }
    __host__ __device__ Vec3f operator*(float s) const { return {x * s, y * s, z * s}; }
    __host__ __device__ Vec3f operator*(const Vec3f &v) const { return {x * v.x, y * v.y, z * v.z}; }
    __host__ __device__ Vec3f operator/(float s) const
    {
        float inv = 1.0f / s;
        return {x * inv, y * inv, z * inv};
    }
    __host__ __device__ Vec3f operator-() const { return {-x, -y, -z}; }

    __host__ __device__ Vec3f &operator+=(const Vec3f &v)
    {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }
    __host__ __device__ Vec3f &operator-=(const Vec3f &v)
    {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }
    __host__ __device__ Vec3f &operator*=(float s)
    {
        x *= s;
        y *= s;
        z *= s;
        return *this;
    }

    __host__ __device__ float operator[](int i) const { return (&x)[i]; }
    __host__ __device__ float &operator[](int i) { return (&x)[i]; }

    __host__ __device__ float length_sq() const { return x * x + y * y + z * z; }
    __host__ __device__ float length() const { return sqrtf(length_sq()); }

    __host__ __device__ Vec3f normalized() const
    {
        float l = length();
        return (l > 1e-12f) ? (*this) * (1.0f / l) : Vec3f(0);
    }
};

__host__ __device__ inline float dot(const Vec3f &a, const Vec3f &b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline Vec3f cross(const Vec3f &a, const Vec3f &b)
{
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

__host__ __device__ inline Vec3f operator*(float s, const Vec3f &v) { return v * s; }
