// y_load.cu
//
// Ghost loading based on threadIdx.y. Requires NX_TILE = NY_TILE.
//

__global__ void
grad_kernel_y_load(const real * __restrict f, real * __restrict u, const real xfactor, const real yfactor,
    const real zfactor)
{
	__shared__ real fs[NY_TILE + 2 * NGHOST][NX_TILE + 2 * NGHOST];
	const int ghostMul[] = { 0, 0, 0, 1, 1, 1 };
	// Local indices
	const int xli = threadIdx.x + NGHOST;
	const int yli = threadIdx.y + NGHOST;
	// Global indices
	const int xi = blockIdx.x * blockDim.x + threadIdx.x + NGHOST;
	const int yi = blockIdx.y * blockDim.y + threadIdx.y + NGHOST;
	// Ghost zone loading indices
	int2 gli = make_int2(-1, -1), gi = make_int2(-1, -1);
	if (threadIdx.y < 2 * NGHOST) {
		int off =  -3 + ghostMul[threadIdx.y] * NY_TILE;
		gli.x = xli;
		gli.y = yli + off;
		gi.x  = xi;
		gi.y  = yi  + off;
	}
	else if (threadIdx.y < 4 * NGHOST) {
		int adjidx = threadIdx.y - 2 * NGHOST;
		int off = -3 + ghostMul[adjidx] * NY_TILE - 2 * NGHOST;
		gli.x = yli + off;
		gli.y = xli;
		gi.x  = blockIdx.x * blockDim.x + yli + off;
		gi.y  = blockIdx.y * blockDim.y + xli;
	}

	// Z-wise iteration values
	real behind3,
	    behind2 =  f[vfidx(xi, yi, 0)],
	    behind1 =  f[vfidx(xi, yi, 1)],
	    current =  f[vfidx(xi, yi, 2)],
	    forward1 = f[vfidx(xi, yi, 3)],
	    forward2 = f[vfidx(xi, yi, 4)],
	    forward3 = f[vfidx(xi, yi, 5)];

	for (int zi = NGHOST; zi < NZ + NGHOST; zi++) {
		// Iterate through z dimension in registers
		behind3 =  behind2;
		behind2 =  behind1;
		behind1 =  current;
		current =  forward1;
		forward1 = forward2;
		forward2 = forward3;
		forward3 = f[vfidx(xi, yi, zi + 3)];

		// Load x-y tile to shared memory
		__syncthreads();
		fs[yli][xli] = current;
		if (gli.x >= 0)
			fs[gli.y][gli.x] = f[vfidx(gi.x, gi.y, zi)];
		__syncthreads();

		// Compute the gradient
		u[vfidx(xi, yi, zi, 2)] = zfactor * fd1D(
		    behind3, behind2, behind1, forward1, forward2, forward3);

		u[vfidx(xi, yi, zi, 1)] = yfactor * fd1D(
		    fs[yli - 3][xli], fs[yli - 2][xli], fs[yli - 1][xli],
		    fs[yli + 1][xli], fs[yli + 2][xli], fs[yli + 3][xli]);

		u[vfidx(xi, yi, zi, 0)] = xfactor * fd1D(
		    fs[yli][xli - 3], fs[yli][xli - 2], fs[yli][xli - 1],
		    fs[yli][xli + 1], fs[yli][xli + 2], fs[yli][xli + 3]);
	}
}

void
grad_y_load(vf3dgpu &f, vf3dgpu &u)
{
	grad_kernel_y_load<<<xy_tile.nblocks, xy_tile.nthreads>>>(f.mem(), u.mem(),
	    1.0/dx, 1.0/dy, 1.0/dz);
}
