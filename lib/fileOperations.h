#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>
#ifndef OPENMP_CUDA_FILEOPERATIONS_H
#define OPENMP_CUDA_FILEOPERATIONS_H

double processFileInChunks(const std::string &inputFile,
                         const std::string &outputFile,
                         ChaCha20Interface *chacha, const uint32_t key[8],
                         const uint32_t nonce[3], uint32_t counter) {
  std::ifstream inFile(inputFile, std::ios::binary);
  std::ofstream outFile(outputFile, std::ios::binary);

  if (!inFile.is_open())
    throw std::runtime_error("Failed to open input file");
  if (!outFile.is_open())
    throw std::runtime_error("Failed to open output file");

  const size_t maxChunkSize = 8ULL * 1024 * 1024 * 1024; // 8 GB
  inFile.seekg(0, std::ios::end);
  size_t fileSize = inFile.tellg();
  inFile.seekg(0, std::ios::beg);

  size_t chunkSize = std::min(fileSize, maxChunkSize);
  std::vector<uint8_t> buffer(chunkSize);

  double totalEncryption = 0.0;

  while (inFile.read(reinterpret_cast<char *>(buffer.data()), chunkSize) ||
         inFile.gcount() > 0) {
    size_t bytesRead = inFile.gcount();
    buffer.resize(bytesRead);

    totalEncryption += chacha->encrypt(buffer.data(), buffer.size(), key, nonce, counter);

    outFile.write(reinterpret_cast<char *>(buffer.data()), bytesRead);

    counter += bytesRead;
  }

  std::cout << "Processing time: " << totalEncryption << " seconds"
            << std::endl;
  std::cout << "Throughput "
            << (fileSize / (1024.0 * 1024.0 * 1024.0)) / totalEncryption
            << " Gb/s" << std::endl;
  std::cout << "File processed successfully." << std::endl;
  inFile.close();
  outFile.close();
  return totalEncryption;
}

#endif // OPENMP_CUDA_FILEOPERATIONS_H
