// boundary.cu
//
// CUDA kernels for applying periodic boundary conditions.
//

__global__ void
periodic_bc_x_kernel(real *f)
{
	int yi = threadIdx.x + blockDim.x * blockIdx.x + NGHOST;
	int zi = threadIdx.y + blockDim.y * blockIdx.y + NGHOST;

	f[vfidx(0, yi, zi)] = f[vfidx(NX + NGHOST - 4, yi, zi)];
	f[vfidx(1, yi, zi)] = f[vfidx(NX + NGHOST - 3, yi, zi)];
	f[vfidx(2, yi, zi)] = f[vfidx(NX + NGHOST - 2, yi, zi)];
	f[vfidx(NX + NGHOST, yi, zi)] = f[vfidx(NGHOST + 1, yi, zi)];
	f[vfidx(NX + NGHOST + 1, yi, zi)] = f[vfidx(NGHOST + 2, yi, zi)];
	f[vfidx(NX + NGHOST + 2, yi, zi)] = f[vfidx(NGHOST + 3, yi, zi)];
}

__global__ void
periodic_bc_y_kernel(real *f)
{
	int xi = threadIdx.x + blockDim.x * blockIdx.x + NGHOST;
	int zi = threadIdx.y + blockDim.y * blockIdx.y + NGHOST;

	f[vfidx(xi, 0, zi)] = f[vfidx(xi, NY + NGHOST - 4, zi)];
	f[vfidx(xi, 1, zi)] = f[vfidx(xi, NY + NGHOST - 3, zi)];
	f[vfidx(xi, 2, zi)] = f[vfidx(xi, NY + NGHOST - 2, zi)];
	f[vfidx(xi, NY + NGHOST, zi)] = f[vfidx(xi, NGHOST + 1, zi)];
	f[vfidx(xi, NY + NGHOST + 1, zi)] = f[vfidx(xi, NGHOST + 2, zi)];
	f[vfidx(xi, NY + NGHOST + 2, zi)] = f[vfidx(xi, NGHOST + 3, zi)];
}

__global__ void
periodic_bc_z_kernel(real *f)
{
	int xi = threadIdx.x + blockDim.x * blockIdx.x + NGHOST;
	int yi = threadIdx.y + blockDim.y * blockIdx.y + NGHOST;

	f[vfidx(xi, yi, 0)] = f[vfidx(xi, yi, NZ + NGHOST - 4)];
	f[vfidx(xi, yi, 1)] = f[vfidx(xi, yi, NZ + NGHOST - 3)];
	f[vfidx(xi, yi, 2)] = f[vfidx(xi, yi, NZ + NGHOST - 2)];
	f[vfidx(xi, yi, NZ + NGHOST)] = f[vfidx(xi, yi, NGHOST + 1)];
	f[vfidx(xi, yi, NZ + NGHOST + 1)] = f[vfidx(xi, yi, NGHOST + 2)];
	f[vfidx(xi, yi, NZ + NGHOST + 2)] = f[vfidx(xi, yi, NGHOST + 3)];
}

void
apply_periodic_bc(vf3dgpu &vf)
{
	dim3 nblocks;
	dim3 nthreads;
	for (int vi = 0; vi < vf.varcount(); vi++) {
		vf3dgpu vfi = vf.subfield(vi, 1);
		nblocks = dim3(1, NZ);
		nthreads = dim3(NY);
		periodic_bc_x_kernel<<<nblocks, nthreads>>>(vfi.mem());
		nblocks = dim3(1, NZ);
		nthreads = dim3(NX);
		periodic_bc_y_kernel<<<nblocks, nthreads>>>(vfi.mem());
		nblocks = dim3(1, NY);
		nthreads = dim3(NX);
		periodic_bc_z_kernel<<<nblocks, nthreads>>>(vfi.mem());
	}
}

