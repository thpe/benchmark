#include <iostream>
#include <chrono>
#include <math.h>
using namespace std;
// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
	  y[i] = x[i] + y[i];
}

__global__
void mul(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] * y[i];
}

int main(void)
{
  int N = 100;
  float *x, *y;
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;

  auto start = std::chrono::system_clock::now ();
  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  auto stop = std::chrono::system_clock::now ();
  chrono::duration< double > dur = stop - start;
  std::cout << "alloc took " << dur.count () << " s " << std::endl;
  start = std::chrono::system_clock::now ();
  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }
  stop = std::chrono::system_clock::now ();
  dur = stop - start;
  std::cout << "init took " << dur.count () << " s " << std::endl;

  auto tstart = std::chrono::system_clock::now ();
  start = std::chrono::system_clock::now ();
  // Run kernel on 1M elements on the GPU
  add<<<numBlocks, blockSize>>>(N, x, y);
  stop = std::chrono::system_clock::now ();
  dur = stop - start;
  std::cout << "add took " << dur.count () << " s " << std::endl;
  start = std::chrono::system_clock::now ();
  // Run kernel on 1M elements on the GPU
  add<<<numBlocks, blockSize>>>(N, x, y);
  stop = std::chrono::system_clock::now ();
  dur = stop - start;
  std::cout << "add took " << dur.count () << " s " << std::endl;
  start = std::chrono::system_clock::now ();
  // Run kernel on 1M elements on the GPU
  add<<<numBlocks, blockSize>>>(N, x, y);
  stop = std::chrono::system_clock::now ();
  dur = stop - start;
  std::cout << "add took " << dur.count () << " s " << std::endl;

  start = std::chrono::system_clock::now ();
  // Run kernel on 1M elements on the GPU
  add<<<numBlocks, blockSize>>>(N, x, y);
  stop = std::chrono::system_clock::now ();
  dur = stop - start;
  std::cout << "add took " << dur.count () << " s " << std::endl;
  start = std::chrono::system_clock::now ();
  // Run kernel on 1M elements on the GPU
  mul<<<numBlocks, blockSize>>>(N, x, y);
  stop = std::chrono::system_clock::now ();
  dur = stop - start;
  std::cout << "mul took " << dur.count () << " s " << std::endl;
  start = std::chrono::system_clock::now ();
  // Run kernel on 1M elements on the GPU
  mul<<<numBlocks, blockSize>>>(N, x, y);
  stop = std::chrono::system_clock::now ();
  dur = stop - start;
  std::cout << "mul took " << dur.count () << " s " << std::endl;
  start = std::chrono::system_clock::now ();
  // Run kernel on 1M elements on the GPU
  mul<<<numBlocks, blockSize>>>(N, x, y);
  stop = std::chrono::system_clock::now ();
  dur = stop - start;
  std::cout << "mul took " << dur.count () << " s " << std::endl;

  start = std::chrono::system_clock::now ();
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  stop = std::chrono::system_clock::now ();
  dur = stop - start;
  std::cout << "sync took " << dur.count () << " s " << std::endl;
  auto tstop = std::chrono::system_clock::now ();
  dur = tstop - tstart;
  std::cout << "total took " << dur.count () << " s " << std::endl;

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
