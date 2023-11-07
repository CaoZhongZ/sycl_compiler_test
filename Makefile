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
SYCLFLAGS=-fsycl -fsycl-id-queries-fit-in-int -fsycl-default-sub-group-size=16 -D__SYCL_INTERNAL_API -fsycl-targets=$(TARGET)
LINKFLAGS=$(OPT_FLAGS) -fsycl -fsycl-max-parallel-link-jobs=8 -fsycl-targets=$(TARGET) $(LINKTARGET)

.PRECIOUS : %-xe.o %.host.o %.dev.bc %.footer.hpp

%.header.hpp %.footer.hpp %.dev.bc : %.cpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(V) -fsycl-device-only -Xclang -fsycl-int-header=$*.header.hpp -Xclang -fsycl-int-footer=$*.footer.hpp -o $*.dev.bc $<

%.host.o : %.header.hpp %.cpp
	$(CXX) $(CXXFLAGS) $(V) -Xclang -fsycl-is-host -D__SYCL_UNNAMED_LAMBDA__ -I$(SYCL_INCLUDE_DIR) -I$(LLVM_INCLUDE_DIR) -c -o $@ -include $^

llvm_base=/opt/intel/oneapi/compiler/2023.1.0/linux

# compatible to different versions
%.o : %.dev.bc %.host.o
	-clang-offload-bundler -type=o -targets=$(OFFLOAD),host-x86_64-unknown-linux-gnu -output=$@ -input=$*.dev.bc,$*.host.o
	-clang-offload-bundler -type=o -targets=$(OFFLOAD),host-x86_64-unknown-linux-gnu -output=$@ -inputs=$*.dev.bc,$*.host.o

%.debug : %.o
	# clang-offload-bundler -type=o -targets=sycl-spir64-unknown-unknown -input=$< -check-section -base-temp-dir=./tmp
	# clang-offload-bundler -type=o -input=$< -list
	clang-offload-bundler -type=o -targets=host-x86_64-unknown-linux-gnu,sycl-spir64_gen-unknown-unknown -input=$< -output=./tmp/$<-x86.o -output=./tmp/$<-xe.o -unbundle -allow-missing-bundles -base-temp-dir=./tmp
	spirv-to-ir-wrapper ./tmp/$<-xe.o -o ./tmp/$<-xe.bc
	llvm-link ./tmp/$<-xe.bc -o ./tmp/$<-xe-linked.bc
	clang-offload-bundler -type=o -targets=sycl-spir64_gen-unknown-unknown -input=$(llvm_base)/lib/libsycl-crt.o -output=./tmp/libsycl-crt.o -unbundle -allow-missing-bundles -base-temp-dir=./tmp
	clang-offload-bundler -type=o -targets=sycl-spir64_gen-unknown-unknown -input=$(llvm_base)/lib/libsycl-complex.o -output=./tmp/libsycl-complex.o -unbundle -allow-missing-bundles -base-temp-dir=./tmp
	clang-offload-bundler -type=o -targets=sycl-spir64_gen-unknown-unknown -input=$(llvm_base)/lib/libsycl-complex-fp64.o -output=./tmp/libsycl-complex-fp64.o -unbundle -allow-missing-bundles -base-temp-dir=./tmp
	clang-offload-bundler -type=o -targets=sycl-spir64_gen-unknown-unknown -input=$(llvm_base)/lib/libsycl-cmath.o -output=./tmp/libsycl-cmath.o -unbundle -allow-missing-bundles -base-temp-dir=./tmp
	clang-offload-bundler -type=o -targets=sycl-spir64_gen-unknown-unknown -input=$(llvm_base)/lib/libsycl-cmath-fp64.o -output=./tmp/libsycl-cmath-fp64.o -unbundle -allow-missing-bundles -base-temp-dir=./tmp
	clang-offload-bundler -type=o -targets=sycl-spir64_gen-unknown-unknown -input=$(llvm_base)/lib/libsycl-imf.o -output=./tmp/libsycl-imf.o -unbundle -allow-missing-bundles -base-temp-dir=./tmp
	clang-offload-bundler -type=o -targets=sycl-spir64_gen-unknown-unknown -input=$(llvm_base)/lib/libsycl-imf-fp64.o -output=./tmp/libsycl-imf-fp64.o -unbundle -allow-missing-bundles -base-temp-dir=./tmp
	clang-offload-bundler -type=o -targets=sycl-spir64_gen-unknown-unknown -input=$(llvm_base)/lib/libsycl-imf-bf16.o -output=./tmp/libsycl-imf-bf16.o -unbundle -allow-missing-bundles -base-temp-dir=./tmp
	clang-offload-bundler -type=o -targets=sycl-spir64_gen-unknown-unknown -input=$(llvm_base)/lib/libsycl-fallback-cassert.o -output=./tmp/libsycl-fallback-cassert.o -unbundle -allow-missing-bundles -base-temp-dir=./tmp
	clang-offload-bundler -type=o -targets=sycl-spir64_gen-unknown-unknown -input=$(llvm_base)/lib/libsycl-fallback-cstring.o -output=./tmp/libsycl-fallback-cstring.o -unbundle -allow-missing-bundles -base-temp-dir=./tmp
	clang-offload-bundler -type=o -targets=sycl-spir64_gen-unknown-unknown -input=$(llvm_base)/lib/libsycl-fallback-complex.o -output=./tmp/libsycl-fallback-complex.o -unbundle -allow-missing-bundles -base-temp-dir=./tmp
	clang-offload-bundler -type=o -targets=sycl-spir64_gen-unknown-unknown -input=$(llvm_base)/lib/libsycl-fallback-complex-fp64.o -output=./tmp/libsycl-fallback-complex-fp64.o -unbundle -allow-missing-bundles -base-temp-dir=./tmp
	clang-offload-bundler -type=o -targets=sycl-spir64_gen-unknown-unknown -input=$(llvm_base)/lib/libsycl-fallback-cmath.o -output=./tmp/libsycl-fallback-cmath.o -unbundle -allow-missing-bundles -base-temp-dir=./tmp
	clang-offload-bundler -type=o -targets=sycl-spir64_gen-unknown-unknown -input=$(llvm_base)/lib/libsycl-fallback-cmath-fp64.o -output=./tmp/libsycl-fallback-cmath-fp64.o -unbundle -allow-missing-bundles -base-temp-dir=./tmp
	clang-offload-bundler -type=o -targets=sycl-spir64_gen-unknown-unknown -input=$(llvm_base)/lib/libsycl-fallback-imf.o -output=./tmp/libsycl-fallback-imf.o -unbundle -allow-missing-bundles -base-temp-dir=./tmp
	clang-offload-bundler -type=o -targets=sycl-spir64_gen-unknown-unknown -input=$(llvm_base)/lib/libsycl-fallback-imf-fp64.o -output=./tmp/libsycl-fallback-imf-fp64.o -unbundle -allow-missing-bundles -base-temp-dir=./tmp
	clang-offload-bundler -type=o -targets=sycl-spir64_gen-unknown-unknown -input=$(llvm_base)/lib/libsycl-fallback-imf-bf16.o -output=./tmp/libsycl-fallback-imf-bf16.o -unbundle -allow-missing-bundles -base-temp-dir=./tmp
	clang-offload-bundler -type=o -targets=sycl-spir64_gen-unknown-unknown -input=$(llvm_base)/lib/libsycl-native-bfloat16.o -output=./tmp/libsycl-native-bfloat16.o -unbundle -allow-missing-bundles -base-temp-dir=./tmp
	clang-offload-bundler -type=o -targets=sycl-spir64_gen-unknown-unknown -input=$(llvm_base)/lib/libsycl-itt-user-wrappers.o -output=./tmp/libsycl-itt-user-wrappers.o -unbundle -allow-missing-bundles -base-temp-dir=./tmp
	clang-offload-bundler -type=o -targets=sycl-spir64_gen-unknown-unknown -input=$(llvm_base)/lib/libsycl-itt-compiler-wrappers.o -output=./tmp/libsycl-itt-compiler-wrappers.o -unbundle -allow-missing-bundles -base-temp-dir=./tmp
	clang-offload-bundler -type=o -targets=sycl-spir64_gen-unknown-unknown -input=$(llvm_base)/lib/libsycl-itt-stubs.o -output=./tmp/libsycl-itt-stubs.o -unbundle -allow-missing-bundles -base-temp-dir=./tmp
	llvm-link ./tmp/$<-xe-linked.bc ./tmp/libsycl-crt.o ./tmp/libsycl-complex.o ./tmp/libsycl-complex-fp64.o ./tmp/libsycl-cmath.o ./tmp/libsycl-cmath-fp64.o ./tmp/libsycl-imf.o ./tmp/libsycl-imf-fp64.o ./tmp/libsycl-imf-bf16.o ./tmp/libsycl-fallback-cassert.o ./tmp/libsycl-fallback-cstring.o ./tmp/libsycl-fallback-complex.o ./tmp/libsycl-fallback-complex-fp64.o ./tmp/libsycl-fallback-cmath.o ./tmp/libsycl-fallback-cmath-fp64.o ./tmp/libsycl-fallback-imf.o ./tmp/libsycl-fallback-imf-fp64.o ./tmp/libsycl-fallback-imf-bf16.o ./tmp/libsycl-native-bfloat16.o ./tmp/libsycl-itt-user-wrappers.o ./tmp/libsycl-itt-compiler-wrappers.o ./tmp/libsycl-itt-stubs.o -o ./tmp/$<-xe-linkall.bc
	sycl-post-link -split=auto -emit-param-info -symbols -emit-exported-symbols -split-esimd -lower-esimd -O3 -spec-const=default -device-globals -o ./tmp/$<.table ./tmp/$<-xe-linkall.bc
	file-table-tform -extract=Code -drop_titles -o ./tmp/$<-list.txt ./tmp/$<.table
	llvm-foreach --in-file-list=./tmp/$<-list.txt --in-replace=./tmp/$<-list.txt --out-ext=spv --out-file-list=./tmp/$<-list1.txt --out-replace=./tmp/$<-list1.txt --out-dir=./tmp --jobs=8 -- llvm-spirv -o ./tmp/$<-list1.txt -spirv-max-version=1.4 -spirv-debug-info-version=ocl-100 -spirv-allow-extra-diexpressions -spirv-allow-unknown-intrinsics=llvm.genx. -spirv-ext=-all,+SPV_EXT_shader_atomic_float_add,+SPV_EXT_shader_atomic_float_min_max,+SPV_KHR_no_integer_wrap_decoration,+SPV_KHR_float_controls,+SPV_KHR_expect_assume,+SPV_KHR_linkonce_odr,+SPV_INTEL_subgroups,+SPV_INTEL_media_block_io,+SPV_INTEL_device_side_avc_motion_estimation,+SPV_INTEL_fpga_loop_controls,+SPV_INTEL_unstructured_loop_controls,+SPV_INTEL_fpga_reg,+SPV_INTEL_blocking_pipes,+SPV_INTEL_function_pointers,+SPV_INTEL_kernel_attributes,+SPV_INTEL_io_pipes,+SPV_INTEL_inline_assembly,+SPV_INTEL_arbitrary_precision_integers,+SPV_INTEL_float_controls2,+SPV_INTEL_vector_compute,+SPV_INTEL_fast_composite,+SPV_INTEL_joint_matrix,+SPV_INTEL_arbitrary_precision_fixed_point,+SPV_INTEL_arbitrary_precision_floating_point,+SPV_INTEL_variable_length_array,+SPV_INTEL_fp_fast_math_mode,+SPV_INTEL_long_constant_composite,+SPV_INTEL_arithmetic_fence,+SPV_INTEL_global_variable_decorations,+SPV_INTEL_task_sequence,+SPV_INTEL_optnone,+SPV_INTEL_token_type,+SPV_INTEL_bfloat16_conversion,+SPV_INTEL_joint_matrix,+SPV_INTEL_hw_thread_queries,+SPV_INTEL_memory_access_aliasing,+SPV_KHR_uniform_group_instructions,+SPV_INTEL_masked_gather_scatter,+SPV_INTEL_tensor_float32_conversion ./tmp/$<-list.txt
	llvm-foreach --out-ext=out --in-file-list=./tmp/$<-list1.txt --in-replace=./tmp/$<-list1.txt  --out-file-list=./tmp/$<.out --out-replace=./tmp/$<.out --jobs=8 -- ocloc -output ./tmp/$<.out -file ./tmp/$<-list1.txt -output_no_suffix -spirv_input -device pvc

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
