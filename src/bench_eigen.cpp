#define EIGEN_STACK_ALLOCATION_LIMIT 0
#include <chrono>
#include <iostream>
#include "eigen_matmul.hpp"

using namespace std;
int main (void)
{
  std::cout << "Eigen benchmark" << std::endl;

  auto start = std::chrono::system_clock::now ();
  MatMul< float, 100, 1000 > mm;
  auto stop = std::chrono::system_clock::now ();

  chrono::duration< double > dur = stop - start;
  std::cout << "init took " << dur.count () << " s " << std::endl;
  start = std::chrono::system_clock::now ();
  mm.run ();
  stop = std::chrono::system_clock::now ();
  dur = stop - start;
  std::cout << "multiplication took " << dur.count () << " s " << std::endl;
  std::cout << mm.res().determinant () << std::endl;
  MatMul< float, 500, 500 > madd;
  start = std::chrono::system_clock::now ();
  madd.add ();
  stop = std::chrono::system_clock::now ();
  dur = stop - start;
  std::cout << "add took " << dur.count () << " s " << std::endl;
  std::cout << madd.res().determinant () << std::endl;



  return 0;
}
