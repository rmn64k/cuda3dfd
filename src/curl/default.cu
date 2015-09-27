// default.cu
//
// Default implementation of curl. Can also be tweaked with launch bounds.
//

// omega = curl(u)
// omega and u are vector fields.
#ifdef CURL_LAUNCH_BOUNDS
__launch_bounds__(NX_TILE*NY_TILE,4)
__global__ void
curl_kernel_lb(const real * __restrict__ u, real * __restrict__ omega, const real xfactor,
    const real yfactor, const real zfactor)
#else
__global__ void
curl_kernel_default(const real * __restrict__ u, real * __restrict__ omega, const real xfactor,
    const real yfactor, const real zfactor)
#endif
{
	__shared__ real us[3][NY_TILE + 2 * NGHOST][NX_TILE + 2 * NGHOST];
	// Local indices
	const int xli = threadIdx.x + NGHOST;
	const int yli = threadIdx.y + NGHOST;
	// Global indices
	const int xi = blockIdx.x * blockDim.x + threadIdx.x + NGHOST;
	const int yi = blockIdx.y * blockDim.y + threadIdx.y + NGHOST;

	// Z-wise iteration values
	real xzbehind3,
	    xzbehind2  = u[vfidx(xi, yi, 0, 0)],
	    xzbehind1  = u[vfidx(xi, yi, 1, 0)],
	    xzcurrent  = u[vfidx(xi, yi, 2, 0)],
	    xzforward1 = u[vfidx(xi, yi, 3, 0)],
	    xzforward2 = u[vfidx(xi, yi, 4, 0)],
	    xzforward3 = u[vfidx(xi, yi, 5, 0)];

	real yzbehind3,
	    yzbehind2  = u[vfidx(xi, yi, 0, 1)],
	    yzbehind1  = u[vfidx(xi, yi, 1, 1)],
	    yzcurrent  = u[vfidx(xi, yi, 2, 1)],
	    yzforward1 = u[vfidx(xi, yi, 3, 1)],
	    yzforward2 = u[vfidx(xi, yi, 4, 1)],
	    yzforward3 = u[vfidx(xi, yi, 5, 1)];

	for (int zi = NGHOST; zi < NZ + NGHOST; zi++) {
		// Iterate through z dimension in registers
		xzbehind3  = xzbehind2;
		xzbehind2  = xzbehind1;
		xzbehind1  = xzcurrent;
		xzcurrent  = xzforward1;
		xzforward1 = xzforward2;
		xzforward2 = xzforward3;
		xzforward3 = u[vfidx(xi, yi, zi + 3, 0)];

		yzbehind3  = yzbehind2;
		yzbehind2  = yzbehind1;
		yzbehind1  = yzcurrent;
		yzcurrent  = yzforward1;
		yzforward1 = yzforward2;
		yzforward2 = yzforward3;
		yzforward3 = u[vfidx(xi, yi, zi + 3, 1)];

		// Load x-y tiles to shared memory
		__syncthreads();
		us[0][yli][xli] = xzcurrent;
		us[1][yli][xli] = yzcurrent;
		us[2][yli][xli] = u[vfidx(xi, yi, zi, 2)];
		if (threadIdx.x < NGHOST) {
			us[1][yli][xli - NGHOST]  = u[vfidx(xi - NGHOST, yi, zi, 1)];
			us[1][yli][xli + NX_TILE] = u[vfidx(xi + NX_TILE, yi, zi, 1)];
			us[2][yli][xli - NGHOST]  = u[vfidx(xi - NGHOST, yi, zi, 2)];
			us[2][yli][xli + NX_TILE] = u[vfidx(xi + NX_TILE, yi, zi, 2)];
		}
		if (threadIdx.y < NGHOST) {
			us[0][yli - NGHOST][xli]  = u[vfidx(xi, yi - NGHOST, zi, 0)];
			us[0][yli + NY_TILE][xli] = u[vfidx(xi, yi + NY_TILE, zi, 0)];
			us[2][yli - NGHOST][xli]  = u[vfidx(xi, yi - NGHOST, zi, 2)];
			us[2][yli + NY_TILE][xli] = u[vfidx(xi, yi + NY_TILE, zi, 2)];
		}
		__syncthreads();

		// Compute the curl
		real d1, d2;

		// zdy - ydz
		d2 = zfactor * fd1D(yzbehind3, yzbehind2, yzbehind1, yzforward1, yzforward2, yzforward3);
		d1 = yfactor * fd1D(us[2][yli - 3][xli], us[2][yli - 2][xli], us[2][yli - 1][xli],
		    us[2][yli + 1][xli], us[2][yli + 2][xli], us[2][yli + 3][xli]);

		omega[vfidx(xi, yi, zi, 0)] = d1 - d2;

		// xdz - zdx
		d1 = zfactor * fd1D(xzbehind3, xzbehind2, xzbehind1, xzforward1, xzforward2, xzforward3);
		d2 = xfactor * fd1D(us[2][yli][xli - 3], us[2][yli][xli - 2], us[2][yli][xli - 1],
		    us[2][yli][xli + 1], us[2][yli][xli + 2], us[2][yli][xli + 3]);

		omega[vfidx(xi, yi, zi, 1)] = d1 - d2;

		// ydx - xdy
		d1 = xfactor * fd1D(us[1][yli][xli - 3], us[1][yli][xli - 2], us[1][yli][xli - 1],
		    us[1][yli][xli + 1], us[1][yli][xli + 2], us[1][yli][xli + 3]);
		d2 = yfactor * fd1D(us[0][yli - 3][xli], us[0][yli - 2][xli], us[0][yli - 1][xli],
		    us[0][yli + 1][xli], us[0][yli + 2][xli], us[0][yli + 3][xli]);

		omega[vfidx(xi, yi, zi, 2)] = d1 - d2;
	}
}

#ifdef CURL_LAUNCH_BOUNDS
void
curl_lb(vf3dgpu &u, vf3dgpu &omega)
{
	curl_kernel_lb<<<xy_tile.nblocks, xy_tile.nthreads>>>(u.mem(), omega.mem(),
	    1.0/dx, 1.0/dy, 1.0/dz);
}
#else
void
curl_default(vf3dgpu &u, vf3dgpu &omega)
{
	curl_kernel_default<<<xy_tile.nblocks, xy_tile.nthreads>>>(u.mem(), omega.mem(),
	    1.0/dx, 1.0/dy, 1.0/dz);
}
#endif
