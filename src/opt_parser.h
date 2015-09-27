#ifndef CUDA_FD_OPT_PARSER_H
#define CUDA_FD_OPT_PARSER_H

#include <string>
#include <vector>

class opt_parser {
private:

	struct optvar {
		enum kind_t { OPT_INT, OPT_DOUBLE, OPT_STR, OPT_TOGGLE } kind;
		std::string fmt;
		union {
			void *vval;
			int *ival;
			double *dval;
			bool *bval;
			std::string *strval;
		};
	};

	std::vector<optvar> opts;

	bool scan_fmt_1(const char *arg, optvar &v);
	bool scan_string(const char *arg, optvar &v);
	bool scan_exact(const char *arg, optvar &v);

public:
	void int_opt(const char *name, int *val);
	void double_opt(const char *name, double *val);
	void string_opt(const char *name, std::string *val);
	void toggle_opt(const char *name, bool *val);

	int parse(int argc, char *argv[]);
};

#endif
