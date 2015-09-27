#ifndef CUDA_FD_VERIFY_OPS_H
#define CUDA_FD_VERIFY_OPS_H

real3 check_grad(vf3dhost &u);
real  check_div(vf3dhost &div);
real3 check_curl(vf3dhost &omega);
real  check_del2(vf3dhost &d2f);
real  check_add2(vf3dhost &b);
real  check_dotmul(vf3dhost &b);

#endif
