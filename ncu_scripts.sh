sudo /usr/local/cuda-12.4/bin/ncu -o profile ./kernel_tests
sudo /usr/local/cuda-12.4/bin/ncu --section MemoryWorkloadAnalysis -f -o profile ./kernel_tests

