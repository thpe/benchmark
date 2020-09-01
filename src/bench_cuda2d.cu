#include <iostream>
#include <chrono>
#include <math.h>
using namespace std;
__global__
void reduce(int n, uint16_t *x, float *y)
{
  int oindex = blockIdx.x * blockDim.x;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
	  y[oindex] = x[i] + y[oindex];
}
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

__global__
void mulscalar(int n, float s, float *x)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    x[i] = x[i] * s;
  }
}

int main(void)
{
  int N = 640 * 480;
  uint16_t *x;
  float *y;
//  dim3 blockSize (16,1,1);
  dim3 numBlocks (40, 30, 1);

  int n = 40 * 30;
  float normfactor = 1.0f / 256.0f;
  
  auto start = std::chrono::system_clock::now ();
  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(uint16_t));
  cudaMallocManaged(&y, n*sizeof(float));

  std::cout << "reduce " << N << " elements " << " to " << n << ".\n";
  auto stop = std::chrono::system_clock::now ();
  chrono::duration< double > dur = stop - start;
  std::cout << "alloc took " << dur.count () << " s " << std::endl;
  start = std::chrono::system_clock::now ();
  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
  }
  for (int i = 0; i < n; i++) {
    y[i] = 0.0f;
  }
  stop = std::chrono::system_clock::now ();
  dur = stop - start;
  std::cout << "init took " << dur.count () << " s " << std::endl;

  auto tstart = std::chrono::system_clock::now ();
  start = std::chrono::system_clock::now ();
  // Run kernel on 1M elements on the GPU
  reduce<<<n, 1>>>(N, x, y);
  mulscalar<<<1, 256>>> (n, normfactor, y);
  stop = std::chrono::system_clock::now ();
  dur = stop - start;
  std::cout << "reduce took " << dur.count () << " s " << std::endl;

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
  for (int i = 0; i < n; i++) {
    maxError = fmax(maxError, fabs(y[i]-1.0f));
  }
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);

  return 0;
}
