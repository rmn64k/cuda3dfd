// basic_ops.cu
//
// CUDA kernels for addition and dot product.
//

__global__ void
dotmul3_kernel(const real * __restrict__ a, const real * __restrict__ b,
    real * __restrict__ c)
{
	int xi = threadIdx.x + blockIdx.x * blockDim.x + NGHOST;
	int yi = threadIdx.y + blockIdx.y * blockDim.y + NGHOST;
	int zi = threadIdx.z + blockIdx.z * blockDim.z + NGHOST;

	int idx1 = vfidx(xi, yi, zi, 0);
	int idx2 = vfidx(xi, yi, zi, 1);
	int idx3 = vfidx(xi, yi, zi, 2);

	c[idx1] = a[idx1] * b[idx1] + a[idx2] * b[idx2] + a[idx3] * b[idx3];
}

void
dotmul3(vf3dgpu &a, vf3dgpu &b, vf3dgpu &c)
{
	dotmul3_kernel<<<x_wide.nblocks, x_wide.nthreads>>>(a.mem(), b.mem(), c.mem());
}

__global__ void
add2_kernel(real c1, const real *t1, real c2, const real *t2, real *res)
{
	int xi = threadIdx.x + blockIdx.x * blockDim.x + NGHOST;
	int yi = threadIdx.y + blockIdx.y * blockDim.y + NGHOST;
	int zi = threadIdx.z + blockIdx.z * blockDim.z + NGHOST;

	int idx = vfidx(xi, yi, zi);
	res[idx] = c1 * t1[idx] + c2 * t2[idx];
}

__global__ void
add3_kernel(real c1, const real *t1, real c2, const real *t2,
    real c3, const real *t3, real *res)
{
	int xi = threadIdx.x + blockIdx.x * blockDim.x + NGHOST;
	int yi = threadIdx.y + blockIdx.y * blockDim.y + NGHOST;
	int zi = threadIdx.z + blockIdx.z * blockDim.z + NGHOST;

	int idx = vfidx(xi, yi, zi);
	res[idx] = c1 * t1[idx] + c2 * t2[idx] + c3 * t3[idx];
}

void
add2(real c1, vf3dgpu &a, real c2, vf3dgpu &b, vf3dgpu &c)
{
	for (int vi = 0; vi < a.varcount(); vi++) {
		add2_kernel<<<x_wide.nblocks, x_wide.nthreads>>>(c1, vfvar(a.mem(), vi),
		    c2, vfvar(b.mem(), vi), vfvar(c.mem(), vi));
	}
}

void
add3(real c1, vf3dgpu &a, real c2, vf3dgpu &b, real c3, vf3dgpu &c, vf3dgpu &d)
{
	for (int vi = 0; vi < a.varcount(); vi++) {
		add3_kernel<<<x_wide.nblocks, x_wide.nthreads>>>(c1, vfvar(a.mem(), vi),
		    c2, vfvar(b.mem(), vi), c3, vfvar(c.mem(), vi), vfvar(d.mem(), vi));
	}
}
