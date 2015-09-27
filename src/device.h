#ifndef CUDA_FD_DEVICE_H
#define CUDA_FD_DEVICE_H

enum init_fun_t {
	TEST_TRIG_INIT,
	SIN_X_INIT,
	ZERO_INIT
};

// Clear GPU memory: mem[0]..mem[n-1] = 0.0;
void clear_gpu_mem(real *mem, int n);

// u = grad(f)
void grad_default(vf3dgpu &f, vf3dgpu &u);
void grad_old_order(vf3dgpu &f, vf3dgpu &u);
void grad_flags(vf3dgpu &f, vf3dgpu &u);
void grad_simple(vf3dgpu &f, vf3dgpu &u);
void grad_noshared(vf3dgpu &f, vf3dgpu &u);
void grad_linear_load(vf3dgpu &f, vf3dgpu &u);
void grad_x_load(vf3dgpu &f, vf3dgpu &u);
void grad_y_load(vf3dgpu &f, vf3dgpu &u);
void grad_three(vf3dgpu &f, vf3dgpu &u);

// f = div(u)
void div_default(vf3dgpu &u, vf3dgpu &f);
void div_same(vf3dgpu &u, vf3dgpu &f);
void div_three(vf3dgpu &u, vf3dgpu &f);

// omega = curl(u)
void curl_default(vf3dgpu &u, vf3dgpu &omega);
void curl_lb(vf3dgpu &u, vf3dgpu &omega);

// d2f = del2(f)
void del2_default(vf3dgpu &f, vf3dgpu &d2f);
void del2_same(vf3dgpu &f, vf3dgpu &d2f);

// c = a . b -- (a and b both of three variables)
void dotmul3(vf3dgpu &a, vf3dgpu &b, vf3dgpu &c);

// c = c1 * a + c2 * b
void add2(real c1, vf3dgpu &a, real c2, vf3dgpu &b, vf3dgpu &c);

// d = c1 * a + c2 * b + c3 * b
void add3(real c1, vf3dgpu &a, real c2, vf3dgpu &b, real c3, vf3dgpu &c, vf3dgpu &d);

// Initialize GPU memory with specified function.
void init_field(vf3dgpu vf, init_fun_t fun, real mod = 1.0);

// Apply periodic boundary conditions
void apply_periodic_bc(vf3dgpu &vf);

// Specialised kernel for the solver
void compute_omega(vf3dgpu u, vf3dgpu omega, int iter, real dt, real visc);

#endif
