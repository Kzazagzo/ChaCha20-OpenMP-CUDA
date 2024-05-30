#ifndef OPENMP_CUDA_CHACHA20_CUDA_CUH
#define OPENMP_CUDA_CHACHA20_CUDA_CUH

#include "../lib/ChaCha20Interface.h"
#include <cstdint>
#include <cstdlib>

namespace ChaChaCuda {
extern __constant__ uint32_t key[8];
extern __constant__ uint32_t nounce[3];

__global__ void chaCha20EncryptKernel(uint8_t *data, size_t length,
                                             uint32_t counter);



class ChaCha20Cuda : public ChaCha20Interface {
public:
  __device__ static inline void quarter_round(uint32_t &a, uint32_t &b,
                                              uint32_t &c, uint32_t &d);
  
  __device__ static inline void chacha20_block(uint32_t output[16],
                                               const uint32_t input[16]);
  
  double encrypt(uint8_t *data, size_t length, const uint32_t key[8],
               const uint32_t nonce[3], uint32_t counter) override;
};
}
#endif // OPENMP_CUDA_CHACHA20_CUDA_CUH
