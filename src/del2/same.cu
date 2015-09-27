// same.cu
//
// Different computation of the Laplacian when using equal spatial discretization.
//

// d2f = del2(f)
// d2f and f are scalar fields.
__global__ void
del2_kernel_same(const real* __restrict__ f, real * __restrict__ d2f, const real factor)
{
	__shared__ real fs[NY_TILE + 2 * NGHOST][NX_TILE + 2 * NGHOST];
	// Local indices
	const int xli = threadIdx.x + NGHOST;
	const int yli = threadIdx.y + NGHOST;
	// Global indices
	const int xi = blockIdx.x * blockDim.x + threadIdx.x + NGHOST;
	const int yi = blockIdx.y * blockDim.y + threadIdx.y + NGHOST;

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
		if (threadIdx.x < NGHOST) {
			fs[yli][xli - NGHOST]  = f[vfidx(xi - NGHOST, yi, zi)];
			fs[yli][xli + NX_TILE] = f[vfidx(xi + NX_TILE, yi, zi)];
		}
		if (threadIdx.y < NGHOST) {
			fs[yli - NGHOST][xli]  = f[vfidx(xi, yi - NGHOST, zi)];
			fs[yli + NY_TILE][xli] = f[vfidx(xi, yi + NY_TILE, zi)];
		}
		__syncthreads();

		// Compute the Laplacian
		real del2Acc = d2Coeff3 * (
		    fs[yli][xli-3] + fs[yli][xli+3] +
		    fs[yli-3][xli] + fs[yli+3][xli] +
		    behind3 + forward3);

		del2Acc -= d2Coeff2 * (
		    fs[yli][xli-2] + fs[yli][xli+2] +
		    fs[yli-2][xli] + fs[yli+2][xli] +
		    behind2 + forward2);

		del2Acc += d2Coeff1 * (
		    fs[yli][xli-1] + fs[yli][xli+1] +
		    fs[yli-1][xli] + fs[yli+1][xli] +
		    behind1 + forward1);

		del2Acc -= d2Coeff0 * 3 * current;

		d2f[vfidx(xi, yi, zi)] = factor * del2Acc;
	}
}

void
del2_same(vf3dgpu &f, vf3dgpu &d2f)
{
	del2_kernel_same<<<xy_tile.nblocks, xy_tile.nthreads>>>(f.mem(), d2f.mem(), 1.0/(dx*dx));
}
