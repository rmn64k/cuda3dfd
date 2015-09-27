#include <cstdio>

#include "common.h"
#include "device.h"

// Variables needed by the CUDA kernels.
// Starting grid point:
real x_0 = 0.0, y_0 = 0.0, z_0 = 0.0;
// Spatial discretization:
real dx = (2 * M_PI - 0.0) / (NX - 1);
real dy = (2 * M_PI - 0.0) / (NY - 1);
real dz = (2 * M_PI - 0.0) / (NZ - 1);

// Choose gradient implementation (see device.h).
#define grad grad_linear_load

int
main()
{
	// Allocate GPU memory for 3D vector fields of one and three variables.
	vf3dgpu f(1), u(3);

	// Allocate host memory for 3D vector field of three variables.
	vf3dhost uh(3);

	// Initialize f to sin(x) + cos(y) - 2 * sin(z)
	init_field(f, TEST_TRIG_INIT);

	// Apply periodic boundary conditions to f.
	apply_periodic_bc(f);

	// Compute the gradient of f and store in u.
	grad(f, u);

	// Copy u GPU memory to the host.
	u.copy_to_host(uh);

	// Get pointer to host memory.
	real *uptr = uh.mem();

	// Write gradients at (x_0 + dx * xi, y_0 + dy, z_0 + 2 * dz)
	for (int xi = 0; xi < NX; xi++) {
		printf("Grad at (%f,%f,%f): (%f,%f,%f).\n",
		    x_0 + dx * xi, y_0 + dy, z_0 + 2 * dz,
		    uptr[vfinsideidx(xi, 1, 2, 0)],
		    uptr[vfinsideidx(xi, 1, 2, 1)],
		    uptr[vfinsideidx(xi, 1, 2, 2)]);
	}

	// Free all memory.
	f.free();
	u.free();
	uh.free();

	return 0;
}
