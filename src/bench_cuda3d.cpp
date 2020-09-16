#include <iostream>
#include <chrono>
#include <math.h>
#include "comp3d.hpp"
#include <Eigen/Eigen>
#include <vector>

using namespace std;

int main(void)
{
 
  int Nf = 256;
  int N = 256 * 40 * 30;
//  float *y;
  float *c;

  std::vector< float > y (30 * 40);
  int n = 40 * 30;
  float normfactor = 1.0f / 256.0f;
  
  auto start = std::chrono::system_clock::now ();
  {
  Comp3d c3d;
  // Allocate Unified Memory – accessible from CPU or GPU

  c3d.alloc (Nf, 30, 40);

  std::cout << "reduce " << N << " elements " << " to " << n << ".\n";
  auto stop = std::chrono::system_clock::now ();
  chrono::duration< double > dur = stop - start;
  std::cout << "alloc took " << dur.count () << " s " << std::endl;
  start = std::chrono::system_clock::now ();
  // initialize x and y arrays on the host
  Eigen::Matrix< uint16_t, Eigen::Dynamic, Eigen::Dynamic > mat (480, 640);
  std::cout << "mat(0,0) " << mat(0,0) << std::endl;
  mat.setOnes();
  std::cout << "mat(0,0) " << mat(0,0) << std::endl;
  c3d.print ();
  c3d.load(mat.data());
  c3d.print ();
  stop = std::chrono::system_clock::now ();
  dur = stop - start;
  std::cout << "init took " << dur.count () << " s " << std::endl;

  auto tstart = std::chrono::system_clock::now ();
  start = std::chrono::system_clock::now ();
  c3d.run ();
  c3d.print ();

  // run
  //
  //
  stop = std::chrono::system_clock::now ();
  dur = stop - start;
  std::cout << "reduce took " << dur.count () << " s " << std::endl;

  start = std::chrono::system_clock::now ();
  // Wait for GPU to finish before accessing on host

  c3d.y (&y[0]);
  dur = stop - start;
  std::cout << "sync took " << dur.count () << " s " << std::endl;
  auto tstop = std::chrono::system_clock::now ();
  dur = tstop - tstart;
  std::cout << "total took " << dur.count () << " s " << std::endl;


  std::cout << "result " << y[0] << ", " << y[1] << ", " << y[2] << std::endl;
  }

  return 0;
}
