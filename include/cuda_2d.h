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


#if 0
__global__
void mulfilt(int n, float *c, float *x, float *y)
{

  int index_x = blockIdx.x * blockDim x + threadIdx.x;
  int index_y = blockIdx.y * blockDim y + threadIdx.y;
  int index_z = blockIdx.z * blockDim z + threadIdx.z;

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    y[i] = x[i] * s;
  }
}
#endif

class Test {
public:
	Test () {}
private:
	int i;
};

