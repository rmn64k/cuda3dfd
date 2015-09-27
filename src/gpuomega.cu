// gpuomega.cu
//
// Specialised kernel for the time step of the solver program.
//

__constant__ real rk32n_alpha[] = { 0.0, -2.0/3.0, -1.0 };

__global__ void
compute_omega_kernel(const real * __restrict__ u, real * __restrict__ omega,
	const int iter, const real dt, const real visc,
    const real inv_dx, const real inv_dy, const real inv_dz,
    const real inv_dx2, const real inv_dy2, const real inv_dz2)
{
	__shared__ real us[NY_TILE + 2 * NGHOST][NX_TILE + 2 * NGHOST];
	// Local indices
	const int xli = threadIdx.x + NGHOST;
	const int yli = threadIdx.y + NGHOST;
	// Global indices
	const int xi = blockIdx.x * blockDim.x + threadIdx.x + NGHOST;
	const int yi = blockIdx.y * blockDim.y + threadIdx.y + NGHOST;

	// Compute for omega_{x,y,z}
	for (int vi = 0; vi < 3; vi++) {
		// Z-wise iteration values
		real behind3,
			behind2  = u[vfidx(xi, yi, 0, vi)],
			behind1  = u[vfidx(xi, yi, 1, vi)],
			current  = u[vfidx(xi, yi, 2, vi)],
			forward1 = u[vfidx(xi, yi, 3, vi)],
			forward2 = u[vfidx(xi, yi, 4, vi)],
			forward3 = u[vfidx(xi, yi, 5, vi)];

		for (int zi = NGHOST; zi < NZ + NGHOST; zi++) {
			// Iterate through z dimension in registers
			behind3  = behind2;
			behind2  = behind1;
			behind1  = current;
			current  = forward1;
			forward1 = forward2;
			forward2 = forward3;
			forward3 = u[vfidx(xi, yi, zi + 3, vi)];

			real ux, uy, uz;
			if (vi == 0) {
				ux = current;
				uy = u[vfidx(xi, yi, zi, 1)];
				uz = u[vfidx(xi, yi, zi, 2)];
			}
			else if (vi == 1) {
				ux = u[vfidx(xi, yi, zi, 0)];
				uy = current;
				uz = u[vfidx(xi, yi, zi, 2)];
			}
			else {
				ux = u[vfidx(xi, yi, zi, 0)];
				uy = u[vfidx(xi, yi, zi, 1)];
				uz = current;
			}

			__syncthreads();
			us[yli][xli] = current;
			if (threadIdx.x < NGHOST) {
				us[yli][xli - NGHOST]  = u[vfidx(xi - NGHOST, yi, zi, vi)];
				us[yli][xli + NX_TILE] = u[vfidx(xi + NX_TILE, yi, zi, vi)];
			}
			if (threadIdx.y < NGHOST) {
				us[yli - NGHOST][xli]  = u[vfidx(xi, yi - NGHOST, zi, vi)];
				us[yli + NY_TILE][xli] = u[vfidx(xi, yi + NY_TILE, zi, vi)];
			}
			__syncthreads();

			real dz_ui = inv_dz * fd1D(
				behind3, behind2, behind1, forward1, forward2, forward3);
			real dy_ui = inv_dy * fd1D(
				us[yli - 3][xli], us[yli - 2][xli], us[yli - 1][xli],
				us[yli + 1][xli], us[yli + 2][xli], us[yli + 3][xli]);
			real dx_ui = inv_dx * fd1D(
				us[yli][xli - 3], us[yli][xli - 2], us[yli][xli - 1],
				us[yli][xli + 1], us[yli][xli + 2], us[yli][xli + 3]);

			real d2_ui = inv_dz2 * fd2D(behind3, behind2, behind1, current,
				forward1, forward2, forward3);
			d2_ui += inv_dy2 * fd2D(
				us[yli - 3][xli], us[yli - 2][xli], us[yli - 1][xli],
				current,
				us[yli + 1][xli], us[yli + 2][xli], us[yli + 3][xli]);
			d2_ui += inv_dx2 * fd2D(
				us[yli][xli - 3], us[yli][xli - 2], us[yli][xli - 1],
				current,
				us[yli][xli + 1], us[yli][xli + 2], us[yli][xli + 3]);

			if (iter > 0) {
				omega[vfidx(xi, yi, zi, vi)] = 
					-dt*(ux*dx_ui + uy*dy_ui + uz*dz_ui) + dt*visc*d2_ui +
				    rk32n_alpha[iter]*omega[vfidx(xi, yi, zi, vi)];
			}
			else {
				omega[vfidx(xi, yi, zi, vi)] = 
					-dt*(ux*dx_ui + uy*dy_ui + uz*dz_ui) + dt*visc*d2_ui;
			}
		}
	}
}

void
compute_omega(vf3dgpu u, vf3dgpu omega, int iter, real dt, real visc)
{ 
	dim3 nblocks(NX / NX_TILE, NY / NY_TILE);
	dim3 nthreads(NX_TILE, NY_TILE);
	compute_omega_kernel<<<nblocks,nthreads>>>(u.mem(), omega.mem(), iter,
	    dt, visc, 1.0/dx, 1.0/dy, 1.0/dz, 1.0/(dx*dx), 1.0/(dy*dy), 1.0/(dz*dz));
}

