// initial.cu
//
// Initialization routines for GPU memory.
//

__device__ real test_trig_init(real x, real y, real z) { return sin(x) + cos(y) - 2 * sin(z); }
__device__ real sin_x_init(real x, real y, real z)     { return sin(x); }
__device__ real zero_init(real x, real y, real z)      { return 0.0; }

template<real(*fun)(real, real, real)>
__global__ void
init_field_kernel(real *f, const real mod, const struct space_args s)
{
	int xi = threadIdx.x + blockDim.x * blockIdx.x;
	int yi = threadIdx.y + blockDim.y * blockIdx.y;
	int zi = threadIdx.z + blockDim.z * blockIdx.z;

	int idx = vfidx(xi + NGHOST, yi + NGHOST, zi + NGHOST);

	f[idx] = mod * fun(s.x_0 + xi * s.dx, s.y_0 + yi * s.dy, s.z_0 + zi * s.dz);
}

void
init_field(vf3dgpu vf, init_fun_t fun, real mod)
{
	space_args s = {dx, dy, dz, x_0, y_0, z_0};
	for (int vi = 0; vi < vf.varcount(); vi++) {
		switch (fun) {
		case TEST_TRIG_INIT:
			init_field_kernel<test_trig_init> <<<x_wide.nblocks, x_wide.nthreads>>>(
			    vfvar(vf.mem(), vi), mod, s);
			break;
		case SIN_X_INIT:
			init_field_kernel<sin_x_init> <<<x_wide.nblocks, x_wide.nthreads>>>(
			    vfvar(vf.mem(), vi), mod, s);
			break;
		case ZERO_INIT:
			init_field_kernel<zero_init> <<<x_wide.nblocks, x_wide.nthreads>>>(
			   vfvar(vf.mem(), vi), mod, s);
			break;
		}
	}
}
