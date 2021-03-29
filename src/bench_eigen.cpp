#define EIGEN_STACK_ALLOCATION_LIMIT 0
#include <chrono>
#include <iostream>

#include "eigen_matmul.hpp"

using namespace std;
int main (void)
{
  std::cout << "Eigen benchmark" << std::endl;

  auto start = std::chrono::system_clock::now ();
  MatMul< float, 100, 100 > mm;
  MatMul< float, 100, 100 > madd;
  MatMul< float, 200, 200 > mm1;
  MatMul< float, 200, 200 > madd1;
  auto stop = std::chrono::system_clock::now ();
  chrono::duration< double > dur = stop - start;
  std::cout << "init took " << dur.count () << " s " << std::endl;

  double dursum = 0.0;
  int runs = 16;

  auto tstart = std::chrono::system_clock::now ();

  for (int i = 0; i < runs; i++) {
    start = std::chrono::system_clock::now ();
    madd.add ();
    stop = std::chrono::system_clock::now ();
    dur = stop - start;
    dursum += dur.count ();
  }
  std::cout << "add took " << dursum / (double)runs << " s in average" << std::endl;

  dursum = 0.0;
  for (int i = 0; i < runs; i++) {
    start = std::chrono::system_clock::now ();
    mm.run ();
    stop = std::chrono::system_clock::now ();
    dur = stop - start;
    dursum += dur.count ();
  }
  std::cout << "multiplication took " << dursum / (double)runs << " s in average" << std::endl;


  dursum = 0.0;
  for (int i = 0; i < runs; i++) {
    start = std::chrono::system_clock::now ();
    madd1.add ();
    stop = std::chrono::system_clock::now ();
    dur = stop - start;
    dursum += dur.count ();
  }
  std::cout << "add took " << dursum / (double)runs << " s in average" << std::endl;

  dursum = 0.0;
  for (int i = 0; i < runs; i++) {
    start = std::chrono::system_clock::now ();
    mm1.run ();
    stop = std::chrono::system_clock::now ();
    dur = stop - start;
    dursum += dur.count ();
  }
  std::cout << "multiplication took " << dursum / (double)runs << " s in average" << std::endl;

  dur = stop - tstart;
  std::cout << "total took " << dur.count () << " s " << std::endl;


  return 0;
}
