// three.cu
//
// Computation of gradient using individual derivatives in different CUDA streams.
//

void
grad_three(vf3dgpu &f, vf3dgpu &u)
{
	real *um = u.mem();
	real *fm = f.mem();
	size_t voff = vfmemsize(1) / sizeof(real);
	cudaStream_t s1, s2;

	check_cuda(cudaStreamCreate(&s1));
	check_cuda(cudaStreamCreate(&s2));

	der_x_kernel<<<xy_tile.nblocks, xy_tile.nthreads>>>(fm, um, 1.0/dx, false);
	der_y_kernel<<<xy_tile.nblocks, xy_tile.nthreads,
	    (NY_TILE+2*NGHOST)*NX_TILE*sizeof(real), s1>>>(fm, &um[voff], 1.0/dy, false);
	der_z_kernel<<<xy_tile.nblocks, xy_tile.nthreads,
	    (NY_TILE+2*NGHOST)*NX_TILE*sizeof(real), s2>>>(fm, &um[voff*2], 1.0/dz, false);

	check_cuda(cudaStreamDestroy(s1));
	check_cuda(cudaStreamDestroy(s2));
}
