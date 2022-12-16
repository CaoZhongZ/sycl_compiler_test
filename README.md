## PyTorch element-wise kernel template sycl toy example for triage compiler problem

Notes:
1. CPU offload element-wise functor to GPU with substantial member size
2. Kernel is 'single shot' style, means no loop.
3. Kernel access member for calculating tensor array strides and dimension edges

## Build
type ```make main``` for compiling whole PyTorch template porting

type ```make small``` for compiling small.cpp

type ```make accessor``` for compiling workaround
