#include <iostream>
#include <chrono>
#include <math.h>
#include <vector>
#include "cuda_2d.h"
#include "cudachk.h"
using namespace std;

int main(void)
{
  float *a;
  float *b;
  float *c;
//  dim3 blockSize (16,1,1);
  dim3 blocks (32, 32, 1);
  dim3 grid (32, 32, 1);

  int n = 8192;
  int m = 1024;

  auto start = std::chrono::system_clock::now ();
  // Allocate Unified Memory – accessible from CPU or GPU
  cudaChk(cudaMalloc (&a, n * m * sizeof(float)));
  cudaChk(cudaMalloc (&b, n * m * sizeof(float)));
  cudaChk(cudaMalloc (&c, m * m * sizeof(float)));


  std::cout << "multiply " << n << " x " << m << ".\n";
  auto stop = std::chrono::system_clock::now ();
  chrono::duration< double > dur = stop - start;
  std::cout << "alloc took " << dur.count () << " s " << std::endl;

  std::vector< float > input (n * m, 4.0f);
  std::vector< float > zero  (n * m, 0.0f);
  std::cout << "size " << input.size () << std::endl;

  start = std::chrono::system_clock::now ();
  // initialize x and y arrays on the host
  cudaChk(cudaMemcpy (a, &input[0], n * m * sizeof(float), cudaMemcpyHostToDevice));
  cudaChk(cudaMemcpy (b, &input[0], n * m * sizeof(float), cudaMemcpyHostToDevice));
  cudaChk(cudaMemcpy (c, &zero [0], m * m * sizeof(float), cudaMemcpyHostToDevice));
  cudaChk(cudaDeviceSynchronize());

  stop = std::chrono::system_clock::now ();
  dur = stop - start;
  std::cout << "init took " << dur.count () << " s " << std::endl;

  auto tstart = std::chrono::system_clock::now ();
  start = std::chrono::system_clock::now ();
  // Run kernel on 1M elements on the GPU
  for (int i = 0; i < 1; i++) {
    matmul<<<grid, blocks>>> (a, b, n, c);
  }
  // Wait for GPU to finish before accessing on host
  cudaChk(cudaDeviceSynchronize());
  stop = std::chrono::system_clock::now ();
  dur = stop - start;
  std::cout << "multiply took " << dur.count () << " s " << std::endl;

  start = std::chrono::system_clock::now ();

  std::vector< float > res (m * m);
  cudaChk(cudaMemcpy (&res[0], c, m * m * sizeof (float), cudaMemcpyDeviceToHost));
  std::cout << "res: " << res[0] << "\n";
  std::cout << "res: " << res[1] << "\n";
  std::cout << "res: " << res[2] << "\n";
  std::cout << "res: " << res[4] << "\n";
  // Free memory
  cudaChk(cudaFree(a));
  cudaChk(cudaFree(b));
  cudaChk(cudaFree(c));

  return 0;
}
