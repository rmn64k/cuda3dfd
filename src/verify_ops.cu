// verify_ops.cu
//
// Tests for the results of the different operators on a simple trigonometric
// function in a 3D volume and peridodic boundaries.
//

#include "common.h"

inline real fun(real x, real y, real z)     { return sin(x) + cos(y) - 2 * sin(z); }
inline real fun_dx(real x, real y, real z)  { return cos(x); }
inline real fun_dy(real x, real y, real z)  { return -sin(y); }
inline real fun_dz(real x, real y, real z)  { return -2 * cos(z); }
inline real fun_d2x(real x, real y, real z) { return -sin(x); }
inline real fun_d2y(real x, real y, real z) { return -cos(y); }
inline real fun_d2z(real x, real y, real z) { return 2 * sin(z); }

real3
check_grad(vf3dhost &u)
{
	real *m = u.mem();
	real3 maxDiff;
	maxDiff.x = 0.0;
	maxDiff.y = 0.0;
	maxDiff.z = 0.0;

	for (int zi = 0; zi < NZ; zi++) {
		for (int yi = 0; yi < NY; yi++) {
			for (int xi = 0; xi < NX; xi++) {
				real x = xi * dx, y = yi * dy, z = zi * dz;
				real diff = fabs(m[vfinsideidx(xi, yi, zi, 0)] - fun_dx(x, y, z));
				if (diff > maxDiff.x) maxDiff.x = diff;
				diff = fabs(m[vfinsideidx(xi, yi, zi, 1)] - fun_dy(x, y, z));
				if (diff > maxDiff.y) maxDiff.y = diff;
				diff = fabs(m[vfinsideidx(xi, yi, zi, 2)] - fun_dz(x, y, z));
				if (diff > maxDiff.z) maxDiff.z = diff;
			}
		}
	}
	return maxDiff;
}

real
check_div(vf3dhost &div)
{
	real *m = div.mem();
	real maxDiff = 0.0;
	for (int zi = 0; zi < NZ; zi++) {
		for (int yi = 0; yi < NY; yi++) {
			for (int xi = 0; xi < NZ; xi++) {
				real x = xi * dx, y = yi * dy, z = zi * dz;
				real target = fun_dx(x, y, z) + fun_dy(x, y, x) + fun_dz(x, y, z);
				real diff = fabs(target - m[vfinsideidx(xi, yi, zi)]);
				if (diff > maxDiff)
					maxDiff = diff;
			}
		}
	}
	return maxDiff;
}

real3
check_curl(vf3dhost &omega)
{
	real *m = omega.mem();
	real3 maxDiff;
	maxDiff.x = 0.0;
	maxDiff.y = 0.0;
	maxDiff.z = 0.0;

	for (int zi = 0; zi < NZ; zi++) {
		for (int yi = 0; yi < NY; yi++) {
			for (int xi = 0; xi < NX; xi++) {
				real x = xi * dx, y = yi * dy, z = zi * dz;
				real diff = fabs(m[vfinsideidx(xi, yi, zi, 0)] - (fun_dy(x, y, z) - fun_dz(x, y, z)));
				if (diff > maxDiff.x) maxDiff.x = diff;
				diff = fabs(m[vfinsideidx(xi, yi, zi, 1)] - (fun_dz(x, y, z) - fun_dx(x, y, z)));
				if (diff > maxDiff.y) maxDiff.y = diff;
				diff = fabs(m[vfinsideidx(xi, yi, zi, 2)] - (fun_dx(x, y, z) - fun_dy(x, y, z)));
				if (diff > maxDiff.z) maxDiff.z = diff;
			}
		}
	}
	return maxDiff;
}

real
check_del2(vf3dhost &d2f)
{
	real *m = d2f.mem();
	real maxDiff = 0.0;
	for (int zi = 0; zi < NZ; zi++) {
		for (int yi = 0; yi < NY; yi++) {
			for (int xi = 0; xi < NZ; xi++) {
				real x = xi * dx, y = yi * dy, z = zi * dz;
				real target = fun_d2x(x, y, z) + fun_d2y(x, y, x) + fun_d2z(x, y, z);
				real diff = fabs(target - m[vfinsideidx(xi, yi, zi)]);
				if (diff > maxDiff)
					maxDiff = diff;
			}
		}
	}
	return maxDiff;
}

real
check_add2(vf3dhost &b)
{
	real *bm = b.mem();
	real maxDiff = 0.0;
	for (int zi = 0; zi < NZ; zi++) {
		for (int yi = 0; yi < NY; yi++) {
			for (int xi = 0; xi < NZ; xi++) {
				real x = xi * dx, y = yi * dy, z = zi * dz;
				real target = 4 * fun(x, y, z);
				real diff = fabs(target - bm[vfinsideidx(xi, yi, zi)]);
				if (diff > maxDiff)
					maxDiff = diff;
			}
		}
	}
	return maxDiff;
}

real
check_dotmul(vf3dhost &b)
{
	real *bm = b.mem();
	real maxDiff = 0.0;
	for (int zi = 0; zi < NZ; zi++) {
		for (int yi = 0; yi < NY; yi++) {
			for (int xi = 0; xi < NZ; xi++) {
				real x = xi * dx, y = yi * dy, z = zi * dz;
				real target = 3*fun(x, y, z)*fun(x, y, z);
				real diff = fabs(target - bm[vfinsideidx(xi, yi, zi)]);
				if (diff > maxDiff)
					maxDiff = diff;
			}
		}
	}
	return maxDiff;
}
