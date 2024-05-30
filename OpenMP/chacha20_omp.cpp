#include "chacha20_omp.h"
#include <chrono>


namespace ChaChaOpenMP {
void quarter_round(uint32_t &a, uint32_t &b, uint32_t &c, uint32_t &d) {
  a += b;
  d ^= a;
  d = (d << 16) | (d >> 16);
  c += d;
  b ^= c;
  b = (b << 12) | (b >> 20);
  a += b;
  d ^= a;
  d = (d << 8) | (d >> 24);
  c += d;
  b ^= c;
  b = (b << 7) | (b >> 25);
}

void chacha20_block(uint32_t *output, const uint32_t *input) {
#pragma omp simd
  for (int i = 0; i != 16; i++) {
    output[i] = input[i];
  }
  for (int i = 0; i != 10; i++) {
    quarter_round(output[0], output[4], output[8], output[12]);
    quarter_round(output[1], output[5], output[9], output[13]);
    quarter_round(output[2], output[6], output[10], output[14]);
    quarter_round(output[3], output[7], output[11], output[15]);
    quarter_round(output[0], output[5], output[10], output[15]);
    quarter_round(output[1], output[6], output[11], output[12]);
    quarter_round(output[2], output[7], output[8], output[13]);
    quarter_round(output[3], output[4], output[9], output[14]);
  }
#pragma omp simd
  for (int i = 0; i != 16; i++) {
    output[i] += input[i];
  }
}

void chaCha20Encrypt(uint8_t *data, size_t length, const uint32_t *key,
                     const uint32_t *nonce, uint32_t counter) {
  size_t numBlocks = (length + 63) / 64;

#pragma omp parallel for num_threads(32)
  for (size_t blockIdx = 0; blockIdx < numBlocks; ++blockIdx) {
    uint32_t state[16] = {0x61707865, 0x3320646e, 0x79622d32, 0x6b206574,
                          key[0], key[1], key[2], key[3],
                          key[4], key[5], key[6], key[7],
                          static_cast<uint32_t>(counter + blockIdx),
                          nonce[0], nonce[1], nonce[2]};

    uint32_t keystream[16];
    uint8_t keystream8[64];

    chacha20_block(keystream, state);
    
#pragma omp simd
    for (int i = 0; i < 16; ++i) {
      ((uint32_t *)keystream8)[i] = keystream[i];
    }

    size_t offset = blockIdx * 64;
    for (size_t i = 0; i < 64 && offset + i < length; ++i) {
      data[offset + i] ^= keystream8[i]; // XOR
    }

    if (++state[12] == 0) {
      ++state[13];
    }
  }
}

double ChaCha20OpenMP::encrypt(uint8_t *data, size_t length,
                               const uint32_t key[8], const uint32_t nonce[3],
                               uint32_t counter) {
  size_t numChunks = (length + chunkSize - 1) / chunkSize;

  double totalTimeSpent = 0.0;

  for (size_t chunkIdx = 0; chunkIdx < numChunks; ++chunkIdx) {
    size_t offset = chunkIdx * chunkSize;
    size_t currentChunkSize =
        (offset + chunkSize <= length) ? chunkSize : (length - offset);

    auto start = std::chrono::high_resolution_clock::now();
    chaCha20Encrypt(data + offset, currentChunkSize, key, nonce,
                    counter + chunkIdx);
    auto end = std::chrono::high_resolution_clock::now();
    totalTimeSpent += std::chrono::duration<double>(end - start).count();
  }
  return totalTimeSpent;
}

} // namespace ChaChaOpenMP
