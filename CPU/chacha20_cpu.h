#ifndef OPENMP_CUDA_CHACHA20_CPU_H
#define OPENMP_CUDA_CHACHA20_CPU_H

#include "../lib/ChaCha20Interface.h"

namespace ChaChaCpu {
class ChaCha20Cpu : public ChaCha20Interface {
public:
  double encrypt(uint8_t *data, size_t length, const uint32_t key[8],
               const uint32_t nonce[3], uint32_t counter) override;
};
}

#endif // OPENMP_CUDA_CHACHA20_CPU_H
