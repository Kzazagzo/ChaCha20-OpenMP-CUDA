#ifndef OPENMP_CUDA_CHACHA20INTERFACE_H
#define OPENMP_CUDA_CHACHA20INTERFACE_H

#include <cstdint>
#include <cstdlib>
#include <omp.h>
#include <cuda_runtime.h>

class ChaCha20Interface {
public:
  int threadsPerBlock = 1024;
  size_t chunkSize = 1024LL * 1024 * 1600; // 64 MB per chunk
  int numStreams = 1;
  
  virtual double encrypt(uint8_t *data, size_t length, const uint32_t key[8],
                       const uint32_t nonce[3], uint32_t counter) = 0;
  virtual ~ChaCha20Interface() = default;
};

#endif // OPENMP_CUDA_CHACHA20INTERFACE_H
