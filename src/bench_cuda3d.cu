#include <iostream>
#include <chrono>
#include <math.h>
#include "comp3d.hpp"
#include <Eigen/Eigen>

using namespace std;

int main(void)
{
 
  int Nf = 256;
  int N = 256 * 40 * 30;
  uint16_t *x;
  float *y;
  float *c;
//  dim3 blockSize (16,1,1);
//  dim3 numBlocks (40, 30, 1);

  int n = 40 * 30;
  float normfactor = 1.0f / 256.0f;
  
  auto start = std::chrono::system_clock::now ();
  {
  Comp3d c3d;
  // Allocate Unified Memory – accessible from CPU or GPU

  c3d.alloc (Nf, 40, 30);

  std::cout << "reduce " << N << " elements " << " to " << n << ".\n";
  auto stop = std::chrono::system_clock::now ();
  chrono::duration< double > dur = stop - start;
  std::cout << "alloc took " << dur.count () << " s " << std::endl;
  start = std::chrono::system_clock::now ();
  // initialize x and y arrays on the host
  Eigen::Matrix< uint16_t, 480, 640 > mat;
  mat.setOnes();
  c3d.load(mat.data());
  stop = std::chrono::system_clock::now ();
  dur = stop - start;
  std::cout << "init took " << dur.count () << " s " << std::endl;

  auto tstart = std::chrono::system_clock::now ();
  start = std::chrono::system_clock::now ();
  // Run kernel on 1M elements on the GPU
  stop = std::chrono::system_clock::now ();
  dur = stop - start;
  std::cout << "reduce took " << dur.count () << " s " << std::endl;

  start = std::chrono::system_clock::now ();
  // Wait for GPU to finish before accessing on host
  dur = stop - start;
  std::cout << "sync took " << dur.count () << " s " << std::endl;
  auto tstop = std::chrono::system_clock::now ();
  dur = tstop - tstart;
  std::cout << "total took " << dur.count () << " s " << std::endl;

  }

  return 0;
}
