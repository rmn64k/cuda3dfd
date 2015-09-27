// noshared.cu
//
// Variant of the gradient not using shared memory.
//

// u = grad(f)
// u is a vector field and f a scalar field.
__global__ void
grad_kernel_noshared(const real* __restrict__ f, real * __restrict__ u, const real xfactor,
    const real yfactor, const real zfactor)
{
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

		// Compute the gradient
		u[vfidx(xi, yi, zi, 0)] = xfactor * fd1D(
		    f[vfidx(xi - 3, yi, zi)],
		    f[vfidx(xi - 2, yi, zi)],
		    f[vfidx(xi - 1, yi, zi)],
		    f[vfidx(xi + 1, yi, zi)],
		    f[vfidx(xi + 2, yi, zi)],
		    f[vfidx(xi + 3, yi, zi)]);

		u[vfidx(xi, yi, zi, 1)] = yfactor * fd1D(
		    f[vfidx(xi, yi - 3, zi)],
		    f[vfidx(xi, yi - 2, zi)],
		    f[vfidx(xi, yi - 1, zi)],
		    f[vfidx(xi, yi + 1, zi)],
		    f[vfidx(xi, yi + 2, zi)],
		    f[vfidx(xi, yi + 3, zi)]);

		u[vfidx(xi, yi, zi, 2)] = zfactor * fd1D(
		    behind3, behind2, behind1,
		    forward1, forward2, forward3);
	}
}

void
grad_noshared(vf3dgpu &f, vf3dgpu &u)
{
	grad_kernel_noshared<<<xy_tile.nblocks, xy_tile.nthreads>>>(f.mem(), u.mem(),
	    1.0/dx, 1.0/dy, 1.0/dz);
}
