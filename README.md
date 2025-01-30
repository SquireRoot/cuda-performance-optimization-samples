# cuda-performance-optimization-samples
Exmple cuda kernels with benchmarking

# Building and running on Linux
To build the project

```bash
mkdir build/
cd build/
cmake .. -DCMAKE_CUDA_ARCHITECTURES=<your cuda architecture>
make
```

For a list of GPUs and their cuda architecture see https://developer.nvidia.com/cuda-gpus

To run the project, go to the build directory and run

```bash
./src/main
```

