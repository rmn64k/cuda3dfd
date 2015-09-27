// simple.cu
//
// Naive implementation of gradient without caching.
//

// u = grad(f)
// u is a vector field and f a scalar field.
__global__ void
grad_kernel_simple(const real* __restrict__ f, real * __restrict__ u, const real xfactor,
    const real yfactor, const real zfactor)
{
	// Global indices
	const int xi = blockIdx.x * blockDim.x + threadIdx.x + NGHOST;
	const int yi = blockIdx.y * blockDim.y + threadIdx.y + NGHOST;
	const int zi = blockIdx.z * blockDim.z + threadIdx.z + NGHOST;

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
	    f[vfidx(xi, yi, zi - 3)],
	    f[vfidx(xi, yi, zi - 2)],
	    f[vfidx(xi, yi, zi - 1)],
	    f[vfidx(xi, yi, zi + 1)],
	    f[vfidx(xi, yi, zi + 2)],
	    f[vfidx(xi, yi, zi + 3)]);
}

void
grad_simple(vf3dgpu &f, vf3dgpu &u)
{
	dim3 nblocks(NX / NX_TILE, NY / NY_TILE, NZ);
	dim3 nthreads(NX_TILE, NY_TILE, 1);
	grad_kernel_simple<<<nblocks, nthreads>>>(f.mem(), u.mem(), 1.0/dx, 1.0/dy, 1.0/dz);
}
