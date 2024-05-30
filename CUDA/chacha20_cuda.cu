#include "chacha20_cuda.cuh"
#include <chrono>

namespace ChaChaCuda {

int calculateOptimalBlocks(int numThreadsPerBlock) {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  int numSMs = prop.multiProcessorCount;

  int maxBlocksPerSM;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &maxBlocksPerSM, ChaChaCuda::chaCha20EncryptKernel, numThreadsPerBlock,
      0);

  int optimalBlocks = maxBlocksPerSM * numSMs;
  return optimalBlocks;
}

__constant__ uint32_t key[8];
__constant__ uint32_t nounce[3];

__global__ void chaCha20EncryptKernel(uint8_t *data, size_t length,
                                      uint32_t counter) {
  uint32_t state[16] = {0x61707865, 0x3320646e, 0x79622d32, 0x6b206574,
                        key[0],     key[1],     key[2],     key[3],
                        key[4],     key[5],     key[6],     key[7],
                        counter,    nounce[0],  nounce[1],  nounce[2]};

  __shared__ uint32_t keystream[16];
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t offset = idx * sizeof(uint4);

  if (offset < length) {
    ChaCha20Cuda::chacha20_block(keystream, state);

    uint4 *data4 = reinterpret_cast<uint4 *>(data + offset);

#pragma unroll
    for (int i = 0; i < 4; ++i) {
      if (offset + i * sizeof(uint4) < length) {
        uint4 value = data4[i];
        value.x ^= keystream[(threadIdx.x * 4 + 0) % 16];
        value.y ^= keystream[(threadIdx.x * 4 + 1) % 16];
        value.z ^= keystream[(threadIdx.x * 4 + 2) % 16];
        value.w ^= keystream[(threadIdx.x * 4 + 3) % 16];
        data4[i] = value;
      }
    }
  }
}

double ChaChaCuda::ChaCha20Cuda::encrypt(uint8_t *data, size_t length,
                                         const uint32_t key[8],
                                         const uint32_t nonce[3],
                                         uint32_t counter) {
  const size_t chunkSize = this->chunkSize;
  const size_t numChunks = (length + chunkSize - 1) / chunkSize;

  uint8_t *cudaData[this->numStreams];
  cudaStream_t streams[this->numStreams];

  for (int i = 0; i < this->numStreams; ++i) {
    cudaMalloc(&cudaData[i], chunkSize);
    cudaStreamCreate(&streams[i]);
  }

  cudaMemcpyToSymbol(ChaChaCuda::key, key, sizeof(uint32_t) * 8);
  cudaMemcpyToSymbol(ChaChaCuda::nounce, nonce, sizeof(uint32_t) * 3);

  // Tail effect fix
  int optimalBlocks = calculateOptimalBlocks(threadsPerBlock);
  double totalTimeSpent = 0.0;
  
  for (size_t chunkIdx = 0; chunkIdx < numChunks; ++chunkIdx) {
    int streamIdx = chunkIdx % this->numStreams;
    size_t offset = chunkIdx * chunkSize;
    size_t currentChunkSize =
        (offset + chunkSize <= length) ? chunkSize : (length - offset);

    cudaMemcpyAsync(cudaData[streamIdx], data + offset, currentChunkSize,
                    cudaMemcpyHostToDevice, streams[streamIdx]);

    
    int numBlocks = optimalBlocks;
    auto start = std::chrono::high_resolution_clock::now();
    ChaChaCuda::chaCha20EncryptKernel<<<numBlocks, this->threadsPerBlock, 0,
                                        streams[streamIdx]>>>(
        cudaData[streamIdx], currentChunkSize, counter + chunkIdx);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    totalTimeSpent += std::chrono::duration<double>(end - start).count();

    cudaMemcpyAsync(data + offset, cudaData[streamIdx], currentChunkSize,
                    cudaMemcpyDeviceToHost, streams[streamIdx]);
  }

  for (int i = 0; i < this->numStreams; ++i) {
    cudaStreamSynchronize(streams[i]);
    cudaFree(cudaData[i]);
    cudaStreamDestroy(streams[i]);
  }
  return totalTimeSpent;
}

__device__ void ChaCha20Cuda::quarter_round(uint32_t &a, uint32_t &b,
                                            uint32_t &c, uint32_t &d) {
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

__device__ void ChaCha20Cuda::chacha20_block(uint32_t *output,
                                             const uint32_t *input) {
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    output[i] = input[i];
  }
  for (int i = 0; i < 10; ++i) {
    quarter_round(output[0], output[4], output[8], output[12]);
    quarter_round(output[1], output[5], output[9], output[13]);
    quarter_round(output[2], output[6], output[10], output[14]);
    quarter_round(output[3], output[7], output[11], output[15]);
    quarter_round(output[0], output[5], output[10], output[15]);
    quarter_round(output[1], output[6], output[11], output[12]);
    quarter_round(output[2], output[7], output[8], output[13]);
    quarter_round(output[3], output[4], output[9], output[14]);
  }
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    output[i] += input[i];
  }
}
} // namespace ChaChaCuda
