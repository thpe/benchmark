#include "cudachk.h"
#include "comp3d.hpp"
#include "cuda_3d.h"
#include <iostream>

struct Comp3d::impl {
  ~impl ();
  int nx_;
  int ny_;
  int nc_;
  bool allocated_;
  uint16_t* in_;
  float*    y_;
  float*    coeff_;
  float*    x_;
  int off_;


  void alloc (int nc, int nx, int ny);
  void load (uint16_t* data);
  void loadCoeff (float* coeff);
  void reduce (int idx);
  void run ();
  void y (float* p);
  void dealloc ();
  void print ();

  void inc_off ();
  int sizeX () const {return nx_ * ny_ * nc_ * sizeof(float);}
  int sizeY () const {return nx_ * ny_ * sizeof (float);}
  int sizeIn() const {return nx_ * ny_ * 16 * 16 * sizeof (uint16_t);}
  int sizeCoeff () const {return nc_ * sizeof (float);}
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
Comp3d::loadCoeff (float* coeff)
{
  d_ptr_->loadCoeff (coeff);
}

void
Comp3d::print ()
{
  d_ptr_->print ();
}

void
Comp3d::impl::alloc (int nc, int nx, int ny)
{
  nx_ = nx;
  ny_ = ny;
  nc_ = nc;
  off_ = 0;
  std::cout << "alloc...";
  cudaChk(cudaMalloc (&x_,     sizeX ()));
  cudaChk(cudaMalloc (&in_,    sizeIn ()));
  cudaChk(cudaMalloc (&y_,     sizeY ()));
  cudaChk(cudaMalloc (&coeff_, sizeCoeff()));
  cudaChk(cudaMemset (x_,     0.0f, sizeX ()));
  cudaChk(cudaMemset (in_,    0u,   sizeIn ()));
  cudaChk(cudaMemset (y_,     0.0f, sizeY ()));
  cudaChk(cudaMemset (coeff_, 0.0f, sizeCoeff()));
  allocated_ = true;
}

void
Comp3d::impl::print ()
{
  print_u16<<< 1, 1 >>> (480, 640, 1, in_);
  print_f32<<< 1, 1 >>> (30, 40, 1, y_);
}

void
Comp3d::impl::loadCoeff (float* coeff)
{
  cudaChk(cudaMemcpy (coeff_, coeff, sizeCoeff(), cudaMemcpyHostToDevice));
}

void
Comp3d::impl::load (uint16_t* data)
{
  cudaChk(cudaMemcpy (in_, data, sizeIn(), cudaMemcpyHostToDevice));
  dim3 gdim (nx_, ny_, 1);
  dim3 bdim (1, 1, 1);
  in_reduce<<< gdim, bdim >>> (nc_, off_, in_, x_);
  inc_off ();
}

void
Comp3d::impl::run (void)
{
  dim3 gdim (1, nx_, ny_);
  dim3 bdim (1, 1, 1);
  filter<<< gdim, bdim >>> (nc_, x_, coeff_, y_);
}

void
Comp3d::impl::inc_off ()
{
  off_++;
  if (off_ >= nc_) {
    off_ = 0;
  }
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
  std::cout << std::hex << y << ", " << y_ << ", " << std::dec <<  sizeY () <<  std::endl;
  cudaChk(cudaMemcpy (y, y_, sizeY (), cudaMemcpyDeviceToHost));
}

void
Comp3d::impl::dealloc ()
{
  std::cout << "dealloc...";
  cudaFree(x_);
  cudaFree(in_);
  cudaFree(y_);
  cudaFree(coeff_);
  allocated_ = false;
  std::cout << "done.\n";
}

Comp3d::impl::~impl ()
{
  dealloc ();
}
