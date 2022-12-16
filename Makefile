CC=clang
CXX=clang++
OPT_FLAG=-O3

ENABLE_AOT=pvc

SYCL_LIB=$(shell clang++ -print-file-name=libsycl.so)
SYCL_ROOT=$(realpath $(dir $(SYCL_LIB)))

SYCL_INCLUDE_DIR=$(SYCL_ROOT)/../include/sycl
LLVM_INCLUDE_DIR=$(SYCL_ROOT)/../include

# Remove the rule
%.o : %.cpp

V=-v
CXXFLAGS=-std=c++17 $(OPT_FLAG) -Wall -Wno-deprecated-declarations -Wno-unused-variable -DSYCL_BUFFER_PARAMS_WRAPPER
SYCLFLAGS=-fsycl -fsycl-id-queries-fit-in-int -fsycl-default-sub-group-size=16 -D__SYCL_INTERNAL_API -fsycl-targets=spir64_gen
SYCLLINK=-fsycl -fsycl-default-sub-group-size=16
LINKFLAGS=-fsycl -fsycl-device-code-split=per_kernel -fsycl-max-parallel-link-jobs=4 -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device $(ENABLE_AOT) -internal_options -ze-intel-has-buffer-offset-arg -internal_options -cl-intel-greater-than-4GB-buffer-required"

.PRECIOUS : %.host.o %.dev.bc %.footer.hpp

%.header.hpp %.footer.hpp %.dev.bc : %.cpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(V) -fsycl-device-only -Xclang -fsycl-int-header=$*.header.hpp -Xclang -fsycl-int-footer=$*.footer.hpp -o $*.dev.bc $<

%.host.o : %.header.hpp %.cpp
	$(CXX) $(CXXFLAGS) $(V) -Xclang -fsycl-is-host -D__SYCL_UNNAMED_LAMBDA__ -I$(SYCL_INCLUDE_DIR) -I$(LLVM_INCLUDE_DIR) -c -o $@ -include $^

%.o : %.dev.bc %.host.o
	clang-offload-bundler -type=o -targets=sycl-spir64_gen-unknown-unknown,host-x86_64-unknown-linux-gnu -outputs=$@ -inputs=$*.dev.bc,$*.host.o

main : main.o
	$(CXX) $(V) $(LINKFLAGS) -o$@ $^

small : small.o
	$(CXX) $(V) $(LINKFLAGS) -o$@ $^

accessor : accessor.o
	$(CXX) $(V) $(LINKFLAGS) -o$@ $^

all : main small accessor

onepass.o : main.cpp
	$(CXX) $(V) $(SYCLFLAGS) -c -o $@ $<

onepass : onepass.o
	$(CXX) $(V) $(LINKFLAGS) -o $@ $<

clean :
	rm -f small main accessor *.o *.header.hpp *.footer.hpp *.bc onepass
