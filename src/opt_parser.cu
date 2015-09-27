#include <cstdio>
#include <string>
#include <vector>

#include "opt_parser.h"

bool
opt_parser::scan_fmt_1(const char *arg, optvar &v)
{
	return sscanf(arg, v.fmt.c_str(), v.vval) == 1;
}

bool
opt_parser::scan_string(const char *arg, optvar &v)
{
	int n = -1;
	sscanf(arg, v.fmt.c_str(), &n);
	if (n > 0) {
		*(v.strval) = std::string(arg + n);
		return true;
	}
	else return false;
}

bool
opt_parser::scan_exact(const char *arg, optvar &v)
{
	if (v.fmt == arg) {
		*(v.bval) = !*(v.bval);
		return true;
	}
	else return false;
}

void
opt_parser::int_opt(const char *name, int *val)
{
	optvar v = { optvar::OPT_INT, std::string(name) + "=%d", val };
	opts.push_back(v);
}

void
opt_parser::double_opt(const char *name, double *val)
{
	optvar v = { optvar::OPT_DOUBLE, std::string(name) + "=%lf", val };
	opts.push_back(v);
}

void
opt_parser::string_opt(const char *name, std::string *val)
{
	optvar v = { optvar::OPT_STR, std::string(name) + "=%n", val };
	opts.push_back(v);
}


void
opt_parser::toggle_opt(const char *name, bool *val)
{
	optvar v = { optvar::OPT_TOGGLE, std::string(name), val };
	opts.push_back(v);
}


int
opt_parser::parse(int argc, char *argv[])
{
	int nopts = opts.size();
	for (int i = 1; i < argc; i++) {
		bool matched = false;
		for (int oi = 0; !matched && oi < nopts; oi++) {
			optvar &opt = opts[oi];
			switch (opt.kind) {
				case optvar::OPT_INT:
				case optvar::OPT_DOUBLE:
					matched = scan_fmt_1(argv[i], opt);
					break;
				case optvar::OPT_STR:
					matched = scan_string(argv[i], opt);
					break;
				case optvar::OPT_TOGGLE:
					matched = scan_exact(argv[i], opt);
					break;
			}
		}
		if (!matched)
			return i;
	}
	return argc;
}

