// grad_flags.cu
//
// Can set which parts of the gradient to compute for with flags.
//

#define GRAD_FLAGS
#define GRAD_X   1
#define GRAD_Y   2
#define GRAD_Z   4
#define GRAD_ALL 7

// u = grad(f)
// u is a vector field and f a scalar field.
__global__ void
grad_kernel_default(const real* __restrict__ f, real * __restrict__ u, const int flags,
    const real xfactor, const real yfactor, const real zfactor)
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
		if ((flags & GRAD_X) && threadIdx.x < NGHOST) {
			fs[yli][xli - NGHOST]  = f[vfidx(xi - NGHOST, yi, zi)];
			fs[yli][xli + NX_TILE] = f[vfidx(xi + NX_TILE, yi, zi)];
		}
		if ((flags & GRAD_Y) && threadIdx.y < NGHOST) {
			fs[yli - NGHOST][xli]  = f[vfidx(xi, yi - NGHOST, zi)];
			fs[yli + NY_TILE][xli] = f[vfidx(xi, yi + NY_TILE, zi)];
		}
		__syncthreads();

		// Compute the gradient
		if (flags & GRAD_Z) {
			u[vfidx(xi, yi, zi, 2)] = zfactor * fd1D(
			    behind3, behind2, behind1, forward1, forward2, forward3);
		}
		if (flags & GRAD_Y) {
			u[vfidx(xi, yi, zi, 1)] = yfactor * fd1D(
			    fs[yli - 3][xli], fs[yli - 2][xli], fs[yli - 1][xli],
			    fs[yli + 1][xli], fs[yli + 2][xli], fs[yli + 3][xli]);
		}
		if (flags & GRAD_X) {
			u[vfidx(xi, yi, zi, 0)] = xfactor * fd1D(
			    fs[yli][xli - 3], fs[yli][xli - 2], fs[yli][xli - 1],
			    fs[yli][xli + 1], fs[yli][xli + 2], fs[yli][xli + 3]);
		}
	}
}

void
grad_flags(vf3dgpu &f, vf3dgpu &u)
{
	grad_kernel_default<<<xy_tile.nblocks, xy_tile.nthreads>>>(f.mem(), u.mem(),
	    GRAD_ALL, 1.0/dx, 1.0/dy, 1.0/dz);
}
