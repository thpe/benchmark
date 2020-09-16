#include "cudachk.h"
#include "comp3d.hpp"
#include "cuda_3d.h"
#include <iostream>

struct Comp3d::impl {
  ~impl ();
  int nx_;
  int ny_;
  int nc_;
  bool allocated;
  struct cudaPitchedPtr in;
  struct cudaPitchedPtr y_;
  struct cudaPitchedPtr coeff;
  struct cudaPitchedPtr x;

  struct cudaExtent in_extent;
  struct cudaExtent y_extent;
  struct cudaExtent coeff_extent;
  struct cudaExtent x_extent;

  cudaMemcpy3DParms in_params;
  cudaMemcpy3DParms out_params;

  void alloc (int nc, int nx, int ny);
  void load (uint16_t* data);
  void reduce (int idx);
  void run ();
  void y (float* p);
  void dealloc ();
  void print ();
};

Comp3d::Comp3d() : 
  d_ptr_(std::make_unique<impl>()) 
{
}


Comp3d::Comp3d(const Comp3d& other) :
  d_ptr_(std::make_unique<impl>(*other.d_ptr_)) 
{
}

Comp3d::Comp3d(Comp3d&& other) = default;

Comp3d& Comp3d::operator=(const Comp3d &other) 
{
  *d_ptr_ = *other.d_ptr_;
  return *this;
}

Comp3d& Comp3d::operator=(Comp3d&&) = default;

Comp3d::~Comp3d() = default;


void
Comp3d::run ()
{
  d_ptr_->run ();
}

void
Comp3d::y (float* p)
{
  std::cout << std::hex << p << std::dec << std::endl; 
  d_ptr_->y (p);
}

void
Comp3d::alloc (int nc, int nx, int ny)
{
  d_ptr_->alloc (nc, nx, ny);
}

void
Comp3d::load (uint16_t* data)
{
  d_ptr_->load (data);
}

void
Comp3d::print ()
{
  d_ptr_->print ();
}

void
Comp3d::impl::alloc (int nc, int nx, int ny)
{
  std::cout << "alloc...";
  x_extent = make_cudaExtent(nc*sizeof(float), nx, ny);
  in_extent = make_cudaExtent(1*sizeof(uint16_t), nx * 16, ny * 16);
  y_extent = make_cudaExtent(nx*sizeof(float), ny, 1);
  coeff_extent = make_cudaExtent(nc*sizeof(float), 1, 1);
  cudaChk(cudaMalloc3D (&x,     x_extent));
  cudaChk(cudaMalloc3D (&in,    in_extent));
  cudaChk(cudaMalloc3D (&y_,    y_extent));
  cudaChk(cudaMalloc3D (&coeff, coeff_extent));
  cudaChk(cudaMemset3D (x,     0.0f, x_extent));
  cudaChk(cudaMemset3D (in,    0u,   in_extent));
  cudaChk(cudaMemset3D (y_,    0.0f, y_extent));
  cudaChk(cudaMemset3D (coeff, 0.0f, coeff_extent));
  allocated = true;

  in_params        = {0};
  in_params.kind   = cudaMemcpyHostToDevice;
  in_params.dstPtr = in;
  in_params.srcPos = make_cudaPos(0,0,0);
  in_params.dstPos = make_cudaPos(0,0,0);
  in_params.extent = in_extent;

  out_params        = {0};
  out_params.kind   = cudaMemcpyDeviceToHost;
  out_params.srcPtr = y_;
  out_params.srcPos = make_cudaPos(0,0,0);
  out_params.dstPos = make_cudaPos(0,0,0);
  out_params.extent = y_extent;

  nx_ = nx;
  ny_ = ny;
  nc_ = nc;
  
  std::cout << "done.\n";
}

void
Comp3d::impl::print ()
{
  print_int<<< 1, 1 >>> (480, 640, 1, in);
  print_int<<< 1, 1 >>> (30, 40, 1, y_);
}

void
Comp3d::impl::load (uint16_t* data)
{
  in_params.srcPtr = make_cudaPitchedPtr(data, 2, 1, 640);
  std::cout << "val " << data[0] << "\n";
  cudaChk(cudaMemcpy3D (&in_params));
}

void
Comp3d::impl::run (void)
{
    dim3 gdim (30, 40, 1);
    dim3 bdim (1, 1, 1);
    in_reduce<<< gdim, bdim >>> (1, 0, (uint16_t*)in.ptr, (float*)y_.ptr);
}


std::ostream& operator<<(std::ostream& os, const cudaExtent& p)
{
    os << p.width << ", " << p.height << ", " << p.depth;
    return os;
}
std::ostream& operator<<(std::ostream& os, const cudaPos& p)
{
    os << p.x << ", " << p.y << ", " << p.z;
    return os;
}
std::ostream& operator<<(std::ostream& os, const cudaPitchedPtr& p)
{
    os << "ptr " << std::hex << p.ptr << std::dec << " pitch: " << p.pitch << ", " << p.xsize << " x " << p.ysize;
    return os;
}
std::ostream& operator<<(std::ostream& os, const cudaMemcpy3DParms& p)
{
    os << p.srcPtr<< " (" << p.srcPos<< ")" <<" -> " << p.dstPtr<< " (" << p.srcPos<< ")" << " ext " << p.extent;
    return os;
}

void
Comp3d::impl::y (float* y)
{
  out_params.dstPtr = make_cudaPitchedPtr(y, 512, nx_ * sizeof(float), ny_);
  std::cout << out_params << std::endl;
  cudaChk(cudaMemcpy3D (&out_params));
}

void
Comp3d::impl::dealloc ()
{
  std::cout << "dealloc...";
  cudaFree(x.ptr);
  cudaFree(in.ptr);
  cudaFree(y_.ptr);
  cudaFree(coeff.ptr);
  allocated = false;
  std::cout << "done.\n";
}

Comp3d::impl::~impl ()
{
  dealloc ();
}
