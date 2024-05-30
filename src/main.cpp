#include "../CPU/chacha20_cpu.h"
#include "../CUDA/chacha20_cuda.cuh"
#include "../OpenMP/chacha20_omp.h"
#include "../lib/fileOperations.h"
#include <chrono>
#include <iostream>
#include <vector>

void parseKeyAndNonce(int argc, char *argv[], uint32_t *key, uint32_t *nonce) {
  if (argc < 14) { // 4 dla nazwy programu, implementacji, pliku wejściowego i wyjściowego + 8 dla klucza + 3 dla nonce
    return;
  }
  
  for (int i = 4; i < 12; ++i) {
    key[i - 4] = std::stoul(argv[i]);
  }
  for (int i = 12; i < 15; ++i) {
    nonce[i - 12] = std::stoul(argv[i]);
  }
}

int main(int argc, char *argv[]) {
  if (argc < 4)
    throw std::runtime_error("Nie podano wystarczającej liczby argumentów do "
                             "uruchomienia programu");
  std::string implementation = "cuda";
  std::string inputFile = "./plik_losowy";
  std::string outputFile = "./out.txt";

  ChaCha20Interface *chacha;
  if (implementation == "cuda") {
    chacha = new ChaChaCuda::ChaCha20Cuda();
  } else if (implementation == "openmp") {
    chacha = new ChaChaOpenMP::ChaCha20OpenMP();
  } else if (implementation == "cpu") {
    chacha = new ChaChaCpu::ChaCha20Cpu();
  } else {
    throw std::runtime_error("Invalid implementation. Use 'cuda' or 'openmp'");
  }

  uint32_t key[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  uint32_t nonce[3] = {0, 1, 2};
  
  parseKeyAndNonce(argc, argv, key, nonce);
  uint32_t counter = 0;

  processFileInChunks(inputFile, outputFile, chacha, key, nonce, counter);


  delete chacha;
  return 0;
}

