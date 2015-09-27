// derivative.cu
//
// Device functions for 6th order FD.
// CUDA kernels for individual derivatives.
//

// Central FD coefficients for 6th order 1st derivatives
__constant__ const real d1Coeff3 =   1.0 / 60.0;
__constant__ const real d1Coeff2 =  -9.0 / 60.0;
__constant__ const real d1Coeff1 =  45.0 / 60.0;

// 6th order 1st derivative central difference stencil.
__device__ real
fd1D(real m3h, real m2h, real m1h, real p1h, real p2h, real p3h)
{
	return d1Coeff3 * (p3h - m3h) +
	    d1Coeff2 * (p2h - m2h) + d1Coeff1 * (p1h - m1h);
}

// Central FD coefficients for 6th order 2nd derivatives
__constant__ const real d2Coeff3 =   2.0 / 180.0;
__constant__ const real d2Coeff2 =  27.0 / 180.0;
__constant__ const real d2Coeff1 = 270.0 / 180.0;
__constant__ const real d2Coeff0 = 490.0 / 180.0;

// 6th order 2nd derivative central difference stencil.
__device__ real
fd2D(real m3h, real m2h, real m1h, real p0h, real p1h, real p2h, real p3h)
{
	return d2Coeff3 * (p3h + m3h) - d2Coeff2 * (p2h + m2h) +
	    d2Coeff1 * (p1h + m1h) - d2Coeff0 * p0h;
}

// CUDA kernels for individual derivatives
__global__ void
der_x_kernel(const real * __restrict__ f, real * __restrict__ df_x, const real xfactor, const bool add)
{
	__shared__ real fs[NY_TILE][NX_TILE + 2 * NGHOST];
	// Local indices
	const int xli = threadIdx.x + NGHOST;
	const int yli = threadIdx.y;
	// Global indices
	const int xi = blockIdx.x * blockDim.x + threadIdx.x + NGHOST;
	const int yi = blockIdx.y * blockDim.y + threadIdx.y + NGHOST;

	for (int zi = NGHOST; zi < NZ + NGHOST; zi++) {
		// Load x-y tile to shared memory
		__syncthreads();
		fs[yli][xli] = f[vfidx(xi, yi, zi)];
		if (threadIdx.x < NGHOST) {
			fs[yli][xli - NGHOST]  = f[vfidx(xi - NGHOST, yi, zi)];
			fs[yli][xli + NX_TILE] = f[vfidx(xi + NX_TILE, yi, zi)];
		}
		__syncthreads();

		real res = xfactor * fd1D(
		    fs[yli][xli - 3], fs[yli][xli - 2], fs[yli][xli - 1],
		    fs[yli][xli + 1], fs[yli][xli + 2], fs[yli][xli + 3]);
		if (add)
			df_x[vfidx(xi, yi, zi)] += res;
		else
			df_x[vfidx(xi, yi, zi)] = res;
	}
}

__global__ void
der_y_kernel(const real * __restrict__ f, real * __restrict__ df_y, const real yfactor, const bool add)
{
	__shared__ real fs[NY_TILE + 2 * NGHOST][NX_TILE];
	// Local indices
	const int xli = threadIdx.x;
	const int yli = threadIdx.y + NGHOST;
	// Global indices
	const int xi = blockIdx.x * blockDim.x + threadIdx.x + NGHOST;
	const int yi = blockIdx.y * blockDim.y + threadIdx.y + NGHOST;

	for (int zi = NGHOST; zi < NZ + NGHOST; zi++) {
		// Load x-y tile to shared memory
		__syncthreads();
		fs[yli][xli] = f[vfidx(xi, yi, zi)];
		if (threadIdx.y < NGHOST) {
			fs[yli - NGHOST][xli]  = f[vfidx(xi, yi - NGHOST, zi)];
			fs[yli + NY_TILE][xli] = f[vfidx(xi, yi + NY_TILE, zi)];
		}
		__syncthreads();

		real res = yfactor * fd1D(
		    fs[yli - 3][xli], fs[yli - 2][xli], fs[yli - 1][xli],
		    fs[yli + 1][xli], fs[yli + 2][xli], fs[yli + 3][xli]);
		if (add)
			df_y[vfidx(xi, yi, zi)] += res;
		else
			df_y[vfidx(xi, yi, zi)] = res;
	}
}

__global__ void
der_z_kernel(const real * __restrict__ f, real * __restrict__ df_z, const real zfactor, const bool add)
{
	__shared__ real fs[NY_TILE + 2 * NGHOST][NX_TILE];
	// Local indices
	const int xli = threadIdx.x;
	const int zli = threadIdx.y + NGHOST;
	// Global indices
	const int xi = blockIdx.x * blockDim.x + threadIdx.x + NGHOST;
	const int zi = blockIdx.y * blockDim.y + threadIdx.y + NGHOST;

	for (int yi = NGHOST; yi < NY + NGHOST; yi++) {
		// Load x-z tile to shared memory
		__syncthreads();
		fs[zli][xli] = f[vfidx(xi, yi, zi)];
		if (threadIdx.y < NGHOST) {
			fs[zli - NGHOST][xli]  = f[vfidx(xi, yi, zi - NGHOST)];
			fs[zli + NY_TILE][xli] = f[vfidx(xi, yi, zi + NY_TILE)];
		}
		__syncthreads();

		real res = zfactor * fd1D(
		    fs[zli - 3][xli], fs[zli - 2][xli], fs[zli - 1][xli],
		    fs[zli + 1][xli], fs[zli + 2][xli], fs[zli + 3][xli]);
		if (add)
			df_z[vfidx(xi, yi, zi)] += res;
		else
			df_z[vfidx(xi, yi, zi)] = res;
	}
}
