// solver.cu
//
// Simulation of Burgers' equation.
// Domain: 0 <= x,y,z <= 2pi
// BC: Periodic
// Initial values: u_x = sin x, u_y=0, u_z = 0
//
// Provides two different solver classes.
// One uses individual kernels, and one
// uses a specialised kernel for this problem.

#include <fstream>

#include "common.h"
#include "device.h"

#include "opt_parser.h"

// Problem values needed by the CUDA kernels.
real x_0 = 0.0;
real y_0 = 0.0;
real z_0 = 0.0;
real dx = (2 * M_PI - 0.0) / (NX - 1);
real dy = (2 * M_PI - 0.0) / (NY - 1);
real dz = (2 * M_PI - 0.0) / (NZ - 1);

// Just call the operators grad and del2.
#define grad grad_default
#define del2 del2_default

struct opts {
	double dt, visc;
	int nts, bs;
	std::string outf;
	bool help;

	opts():
		dt(0.0025),
		visc(0.02),
		nts(100),
		bs(1),
		outf(""),
		help(false)
	{ }

	void usage() {
		puts("Usage: solver [opt=val | help] ...\n\n"
		       "Options with default setting:\n"
			   "dt=0.0025\n"
			   "visc=0.02\n"
			   "sim=1                     (1 modular ops, 2 omega-op)\n"
			   "nts=100                   (number of timesteps to run)\n"
			   "out=                      (file to save to, or empty for none)\n"
			   "\n"
			   "help                      (show this message)\n");
		exit(1);
	}

	void parse(int argc, char *argv[]) {
		int pstop;

		opt_parser op;
		op.double_opt("dt", &dt);
		op.double_opt("visc", &visc);
		op.int_opt("nts", &nts);
		op.int_opt("sim", &bs);
		op.string_opt("out", &outf);
		op.toggle_opt("help", &help);

		if ((pstop = op.parse(argc, argv)) != argc) {
			printf("Don't know the meaning of: '%s'\n\n", argv[pstop]);
			usage();
		}
		if (help)
			usage();
	}
};

// Burgers' simulation using the individual operators.
struct burgsim1 {
	vf3dgpu u, omega, g_ui, u_g_ui, d2_ui;
	real dt, visc;

	burgsim1(opts &opts):
	    u(3), omega(3), g_ui(3), u_g_ui(1), d2_ui(1), dt(opts.dt), visc(opts.visc)
	{
		reset();
	}

	~burgsim1() {
		u.free();
		omega.free();
		g_ui.free();
		u_g_ui.free();
		d2_ui.free();
	}

	void reset() {
		init_field(u.subfield(0, 1), SIN_X_INIT);
		init_field(u.subfield(1, 2), ZERO_INIT);
		apply_periodic_bc(u);
	}

	// RK3-but not quite 2N :)
	void timestep()
	{
		static const real alpha[] = { 0.0, -2.0/3.0, -1.0 };
		static const real beta[] = { 1.0/3.0, 1.0, 1.0/2.0 };

		for (int i = 0; i < 3; i++) {
			for (int vi = 0; vi < 3; vi++) {
				vf3dgpu ui = u.subfield(vi, 1);
				vf3dgpu omegai = omega.subfield(vi, 1);
				grad(ui, g_ui);
				del2(ui, d2_ui);
				dotmul3(u, g_ui, u_g_ui);
				if (i > 0)
					add3(-dt, u_g_ui, dt*visc, d2_ui, alpha[i], omegai, omegai);
				else
					add2(-dt, u_g_ui, dt*visc, d2_ui, omegai);
			}
			add2(1, u, beta[i], omega, u);
			apply_periodic_bc(u);
		}
	}
};

// Burgers' simulation using the specialised kernel.
struct burgsim2 {
	vf3dgpu u, omega;
	real dt, visc;

	burgsim2(opts &opts):
	    u(3), omega(3), dt(opts.dt), visc(opts.visc)
	{
		reset();
	}

	~burgsim2() {
		u.free();
		omega.free();
	}

	void reset() {
		init_field(u.subfield(0, 1), SIN_X_INIT);
		init_field(u.subfield(1, 2), ZERO_INIT);
		apply_periodic_bc(u);
	}

	// RK3-2N :)
	void timestep()
	{
		static const real beta[] = { 1.0/3.0, 1.0, 1.0/2.0 };

		for (int i = 0; i < 3; i++) {
			compute_omega(u, omega, i, dt, visc);
			add2(1, u, beta[i], omega, u);
			apply_periodic_bc(u);
		}
	}
};


// Write a line from u_x.
void
write_xs(const char *name, vf3dgpu u)
{
	vf3dhost uxh(1);
	u.subfield(0, 1).copy_to_host(uxh);
	real *f = uxh.mem();
	int yi = NY / 2, zi = NZ / 2;
	std::ofstream out(name);
	out << "# X\tY" << std::endl;
	for (int xi = 0; xi < NX; xi++)
		out << xi * dx << "\t" << f[vfidx(xi + NGHOST, yi, zi)] << std::endl;
	out.close();
	uxh.free();
}

// Make sure the GPU isn't idling.
template<typename BS> void
warm_up(BS &bs)
{
	puts("Warm up.");
	double last, now, dur;
	dur = 0.0;
	now = read_time_ms();
	while (dur < 2000.0) {
		last = now;
		bs.timestep();
		now = read_time_ms();
		dur += now - last;
	}
	bs.reset();
	puts("Done.");
}

// Run the simulation.
template <typename BS> void
run_bs(opts &opts)
{
	BS bs(opts);
	warm_up(bs);
	double start = read_time_ms();
	for (int i = 0; i < opts.nts; i++)
		bs.timestep();
	double stop = read_time_ms();
	printf("Compute time: %f ms, avg: %f ms, %E us per grid point.\n",
	    stop - start, (stop - start)/opts.nts,
		((stop - start)*1000.0)/((double)opts.nts*NX*NY*NZ));
	if (opts.outf.size() > 0)
		write_xs(opts.outf.c_str(), bs.u);
}

int
main(int argc, char *argv[])
{
	opts opts;
	opts.parse(argc, argv);

	switch (opts.bs) {
	case 1:
		run_bs<burgsim1>(opts);
		break;
	case 2:
		run_bs<burgsim2>(opts);
		break;
	default:
		puts("No such simulation.");
		exit(1);
	}
	puts("Done!");
	return 0;
}
