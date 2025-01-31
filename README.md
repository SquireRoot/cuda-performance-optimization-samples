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

To run the benchmarks, go to the build directory and run

```bash
./src/main-benchmark
```

To run the kernel tests

```bash
./src/main-test
```

# Results

All benchmarks below were performed on my laptop which has the following specs

### nvidia-smi
```
Fri Jan 31 15:31:41 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.144.03             Driver Version: 550.144.03     CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA RTX 500 Ada Gener...    Off |   00000000:01:00.0 Off |                  N/A |
| N/A   40C    P0            364W /   35W |       7MiB /   4094MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A      4903      G   /usr/lib/xorg/Xorg                              4MiB |
+-----------------------------------------------------------------------------------------+
```

### lscpu
```
Architecture:             x86_64
  CPU op-mode(s):         32-bit, 64-bit
  Address sizes:          46 bits physical, 48 bits virtual
  Byte Order:             Little Endian
CPU(s):                   18
  On-line CPU(s) list:    0-17
Vendor ID:                GenuineIntel
  Model name:             Intel(R) Core(TM) Ultra 5 125H
    CPU family:           6
    Model:                170
    Thread(s) per core:   2
    Core(s) per socket:   14
    Socket(s):            1
    Stepping:             4
    CPU(s) scaling MHz:   43%
    CPU max MHz:          4500.0000
    CPU min MHz:          400.0000
...
```

### Output of ./src/main-benchmark

```
## CuBlasMatMultBenchmark

### [0] NVIDIA RTX 500 Ada Generation Laptop GPU

| Samples |  CPU Time  | Noise |  GPU Time  | Noise | Samples | Batch GPU  |
|---------|------------|-------|------------|-------|---------|------------|
|   1280x | 652.827 us | 5.44% | 644.528 us | 5.25% |   1281x | 583.363 us |

## BaselineMatMultBenchmark

### [0] NVIDIA RTX 500 Ada Generation Laptop GPU

| Samples | CPU Time | Noise | GPU Time | Noise | Samples | Batch GPU |
|---------|----------|-------|----------|-------|---------|-----------|
|     66x | 7.627 ms | 0.19% | 7.619 ms | 0.15% |     75x |  7.074 ms |

## MatMult_1_Benchmark

### [0] NVIDIA RTX 500 Ada Generation Laptop GPU

| Samples | CPU Time | Noise | GPU Time | Noise | Samples | Batch GPU |
|---------|----------|-------|----------|-------|---------|-----------|
|   2640x | 2.851 ms | 0.82% | 2.843 ms | 0.77% |   2641x |  2.741 ms |
```