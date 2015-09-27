// three.cu
//
// Use the individual derivates to compute the divergence.
//

void
div_three(vf3dgpu &u, vf3dgpu &f)
{
	real *um = u.mem();
	real *fm = f.mem();
	size_t voff = vfmemsize(1) / sizeof(real);

	der_x_kernel<<<xy_tile.nblocks, xy_tile.nthreads>>>(um, fm, 1.0/dx, false);
	der_y_kernel<<<xy_tile.nblocks, xy_tile.nthreads>>>(&um[voff], fm, 1.0/dy, true);
	der_z_kernel<<<xy_tile.nblocks, xy_tile.nthreads>>>(&um[voff*2], fm, 1.0/dz, true);
}
