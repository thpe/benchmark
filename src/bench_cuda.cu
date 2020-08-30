#include <iostream>
#include <chrono>
#include <math.h>
using namespace std;
// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
    y[i] = x[i] + y[i];
}

int main(void)
{
  int N = 1<<20;
  float *x, *y;

  auto start = std::chrono::system_clock::now ();
  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }
  auto stop = std::chrono::system_clock::now ();
  chrono::duration< double > dur = stop - start;
  std::cout << "init took " << dur.count () << " s " << std::endl;

  start = std::chrono::system_clock::now ();
  // Run kernel on 1M elements on the GPU
  add<<<1, 1>>>(N, x, y);
  stop = std::chrono::system_clock::now ();
  dur = stop - start;
  std::cout << "add took " << dur.count () << " s " << std::endl;

  start = std::chrono::system_clock::now ();
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  stop = std::chrono::system_clock::now ();
  dur = stop - start;
  std::cout << "sync took " << dur.count () << " s " << std::endl;

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);

  return 0;
}
