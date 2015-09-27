// same.cu
//
// Different computation of gradient for equal spatial discretization.
//

// f = div(u)
// f is a scalar field and u a vector field.
__global__ void
div_kernel_same(const real * __restrict__ u, real * __restrict__ f, const real factor)
{
	__shared__ real us[2][NY_TILE + 2 * NGHOST][NX_TILE + 2 * NGHOST];
	// Local indices
	const int xli = threadIdx.x + NGHOST;
	const int yli = threadIdx.y + NGHOST;
	// Global indices
	const int xi = blockIdx.x * blockDim.x + threadIdx.x + NGHOST;
	const int yi = blockIdx.y * blockDim.y + threadIdx.y + NGHOST;

	// Z-wise iteration values
	real zbehind3,
	    zbehind2  = u[vfidx(xi, yi, 0, 2)],
	    zbehind1  = u[vfidx(xi, yi, 1, 2)],
	    zcurrent  = u[vfidx(xi, yi, 2, 2)],
	    zforward1 = u[vfidx(xi, yi, 3, 2)],
	    zforward2 = u[vfidx(xi, yi, 4, 2)],
	    zforward3 = u[vfidx(xi, yi, 5, 2)];

	for (int zi = NGHOST; zi < NZ + NGHOST; zi++) {
		// Iterate through z dimension in registers
		zbehind3  = zbehind2;
		zbehind2  = zbehind1;
		zbehind1  = zcurrent;
		zcurrent  = zforward1;
		zforward1 = zforward2;
		zforward2 = zforward3;
		zforward3 = u[vfidx(xi, yi, zi + 3, 2)];

		// Load x-y tiles to shared memory
		__syncthreads();
		us[0][yli][xli] = u[vfidx(xi, yi, zi, 0)];
		if (threadIdx.x < NGHOST) {
			us[0][yli][xli - NGHOST]  = u[vfidx(xi - NGHOST, yi, zi, 0)];
			us[0][yli][xli + NX_TILE] = u[vfidx(xi + NX_TILE, yi, zi, 0)];
		}
		us[1][yli][xli] = u[vfidx(xi, yi, zi, 1)];
		if (threadIdx.y < NGHOST) {
			us[1][yli - NGHOST][xli]  = u[vfidx(xi, yi - NGHOST, zi, 1)];
			us[1][yli + NY_TILE][xli] = u[vfidx(xi, yi + NY_TILE, zi, 1)];
		}
		__syncthreads();

		// Compute the divergence
		real divAcc = d1Coeff1 * (
		    us[0][yli][xli+1] - us[0][yli][xli-1] +
		    us[1][yli+1][xli] - us[1][yli-1][xli] +
		    zforward1 - zbehind1
		 );
		 divAcc += d1Coeff2 * (
		    us[0][yli][xli+2] - us[0][yli][xli-2] +
		    us[1][yli+2][xli] - us[1][yli-2][xli] +
		    zforward2 - zbehind2
		 );
		 divAcc += d1Coeff3 * (
		    us[0][yli][xli+3] - us[0][yli][xli-3] +
		    us[1][yli+3][xli] - us[1][yli-3][xli] +
		    zforward3 - zbehind3
		 );
		f[vfidx(xi, yi, zi)] = factor * divAcc;
	}
}

void
div_same(vf3dgpu &u, vf3dgpu &f)
{
	div_kernel_same<<<xy_tile.nblocks, xy_tile.nthreads>>>(u.mem(), f.mem(), 1.0/dx);
}
