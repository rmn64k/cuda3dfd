// linear_load.cu
//
// Linear loading of ghost zones.
//

__device__ int2
compute_ghost_index(int idx)
{
	const int nxghost = NGHOST * NX_TILE, nyghost = NGHOST * NY_TILE;
	const int first = nxghost, second = first + nyghost, third = second + nyghost,
	    fourth = third + nxghost;
	if (idx < first)
		return make_int2(NGHOST + idx % NX_TILE, idx / NX_TILE);
	else if (idx < second)
		return make_int2((idx - first) % NGHOST, NGHOST  +  (idx - first) / NGHOST);
	else if (idx < third)
		return make_int2(NGHOST + NX_TILE + (idx - second) % NGHOST,
		    NGHOST + (idx - second) / NGHOST);
	else if (idx < fourth)
	    return make_int2(NGHOST + (idx - third) % NY_TILE,
		    NGHOST + NY_TILE  + (idx - third) / NY_TILE);
	else
		return make_int2(-1, -1);
}

// linear ghost cell loading
__global__ void
grad_kernel_linear_load(const real * __restrict f, real * __restrict u, const real xfactor, const real yfactor,
    const real zfactor)
{
	__shared__ real fs[NY_TILE + 2 * NGHOST][NX_TILE + 2 * NGHOST];
	// Local indices
	const int xli = threadIdx.x + NGHOST;
	const int yli = threadIdx.y + NGHOST;
	// Global indices
	const int xi = blockIdx.x * blockDim.x + threadIdx.x + NGHOST;
	const int yi = blockIdx.y * blockDim.y + threadIdx.y + NGHOST;

	// Ghost zone loading indices
	const int2 gli = compute_ghost_index(threadIdx.y * blockDim.x + threadIdx.x);
	const int2 gi = make_int2(xi - xli + gli.x, yi - yli + gli.y);

	// Z-wise iteration values
	real behind3,
	    behind2  = f[vfidx(xi, yi, 0)],
	    behind1  = f[vfidx(xi, yi, 1)],
	    current  = f[vfidx(xi, yi, 2)],
	    forward1 = f[vfidx(xi, yi, 3)],
	    forward2 = f[vfidx(xi, yi, 4)],
	    forward3 = f[vfidx(xi, yi, 5)];

	for (int zi = NGHOST; zi < NZ + NGHOST; zi++) {
		// Iterate through z dimension in registers
		behind3  = behind2;
		behind2  = behind1;
		behind1  = current;
		current  = forward1;
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
grad_linear_load(vf3dgpu &f, vf3dgpu &u)
{
	grad_kernel_linear_load<<<xy_tile.nblocks, xy_tile.nthreads>>>(f.mem(), u.mem(),
	    1.0/dx, 1.0/dy, 1.0/dz);
}
