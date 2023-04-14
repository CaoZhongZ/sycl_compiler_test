CC=clang
CXX=clang++
OPT_FLAGS=-O3 -ffast-math

ENABLE_AOT=pvc

SYCL_LIB=$(shell clang++ -print-file-name=libsycl.so)
SYCL_ROOT=$(realpath $(dir $(SYCL_LIB)))

SYCL_INCLUDE_DIR=$(SYCL_ROOT)/../include/sycl
LLVM_INCLUDE_DIR=$(SYCL_ROOT)/../include

# Remove the rule
%.o : %.cpp

# default to SYCL
BACKEND?=SYCL

ifeq ($(BACKEND),CUDA)
TARGET=nvptx64-nvidia-cuda
LINKTARGET=-Xsycl-target-backend=nvptx64-nvidia-cuda '--cuda-gpu-arch=sm_86'
OFFLOAD=sycl-nvptx64-nvidia-cuda
else
TARGET=spir64_gen
LINKTARGET=-Xsycl-target-backend=spir64_gen "-device $(ENABLE_AOT) -internal_options -ze-intel-has-buffer-offset-arg -internal_options -cl-intel-greater-than-4GB-buffer-required"
OFFLOAD=sycl-spir64_gen-unknown-unknown
endif

V=-v
CXXFLAGS=-std=c++17 $(OPT_FLAGS) -Wall -Wno-deprecated-declarations -Wno-unused-variable
SYCLFLAGS=-fsycl -fsycl-id-queries-fit-in-int -fsycl-default-sub-group-size=32 -D__SYCL_INTERNAL_API -fsycl-targets=$(TARGET)
LINKFLAGS=$(OPT_FLAGS) -fsycl -fsycl-max-parallel-link-jobs=8 -fsycl-targets=$(TARGET) $(LINKTARGET)

.PRECIOUS : %.host.o %.dev.bc %.footer.hpp

%.header.hpp %.footer.hpp %.dev.bc : %.cpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(V) -fsycl-device-only -Xclang -fsycl-int-header=$*.header.hpp -Xclang -fsycl-int-footer=$*.footer.hpp -o $*.dev.bc $<

%.host.o : %.header.hpp %.cpp
	$(CXX) $(CXXFLAGS) $(V) -Xclang -fsycl-is-host -D__SYCL_UNNAMED_LAMBDA__ -I$(SYCL_INCLUDE_DIR) -I$(LLVM_INCLUDE_DIR) -c -o $@ -include $^

# compatible to different versions
%.o : %.dev.bc %.host.o
	-clang-offload-bundler -type=o -targets=$(OFFLOAD),host-x86_64-unknown-linux-gnu -output=$@ -input=$*.dev.bc,$*.host.o
	-clang-offload-bundler -type=o -targets=$(OFFLOAD),host-x86_64-unknown-linux-gnu -output=$@ -inputs=$*.dev.bc,$*.host.o

main : softmax.o main.o
	$(CXX) $(V) $(LINKFLAGS) -o$@ $^

all : main

clean :
	rm -f main *.o *.header.hpp *.footer.hpp *.bc
