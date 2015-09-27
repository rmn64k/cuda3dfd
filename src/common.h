// common.h
//
// Common declarations and headers.
//

#ifndef CUDA_FD_COMMON_H
#define CUDA_FD_COMMON_H

#include <time.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "errno.h"

#if (NX) % (NX_TILE) != 0
#error NX_TILE must be a divisor of NX
#endif
#if (NY) % (NY_TILE) != 0
#error NY_TILE must be a divisor of NY
#endif

#define NGHOST 3

#if !defined(SINGLE_PRECISION)
typedef double  real;
typedef double3 real3;
#else
typedef float   real;
typedef float3  real3;
#endif

#define check_cuda(err) \
do { \
	cudaError_t cErr = (err); \
	if (cErr != cudaSuccess) { \
		printf("Error: %s.\n", cudaGetErrorString(cErr)); \
		exit(1); \
	} \
} while (0)

inline double
read_time_ms()
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
	return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6; // milliseconds
}

inline long
time_resolution_ns()
{
	struct timespec ts;
	if (clock_getres(CLOCK_MONOTONIC_RAW, &ts) != 0) {
		printf("Error reading clock resolution: '%s'.\n", strerror(errno));
		exit(1);
	}
	return ts.tv_nsec;
}

#include "field3.h"

// These variables need to be provided by the solver.
// dx, dy, dz    - grid spacing.
// x_0, y_0, z_0 - start grid point.
extern real dx, dy, dz, x_0, y_0, z_0;

struct space_args {
	real dx, dy, dz, x_0, y_0, z_0;
};

#endif
