## PyTorch kernel templates sycl example for triage compiler behaviors

Notes:
1. CPU offload functor to GPU with substantial member size
2. Kernel is 'single shot' style, means no loop.
3. Kernel access member for calculating tensor array strides and dimension edges

## Build System Overview
Makefile dissects SYCL compiler into multiple stages.
1. Object stage, compile source files into offloading byte-code and host objects, generating integration headers
2. Bundle both objects and byte-codes into single object
3. Debug device compilation process by ```make <appname>.debug```, it'll generate linking and device compilation process for IGC commands
