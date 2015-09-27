#!/bin/sh

FLAGS="-O3 -r8 -Mcuda=cc2+,ptxinfo,cuda7.0,noL1"
FC=pgf90

${FC} ${FLAGS} -c params.cuf
${FC} ${FLAGS} -c fdops.cuf
${FC} ${FLAGS} -c testfun.cuf
${FC} ${FLAGS} -o testgrad testgrad.cuf params.o fdops.o testfun.o
${FC} ${FLAGS} -o testdiv testdiv.cuf params.o fdops.o testfun.o
${FC} ${FLAGS} -o testcurl testcurl.cuf params.o fdops.o testfun.o

