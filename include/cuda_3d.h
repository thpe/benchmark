__global__
void print_f32 (int nx, int ny, int nz, float* x)
{

    printf("d_tensor[%d][%d][%d] = %f\n", nx, ny, nz, x[nx + gridDim.x * blockDim.x * ny + gridDim.x * blockDim.x * gridDim.y * blockDim.y * nz]);
#if 0
   for (i = 0; i < nx; i++) {
      for (j = 0; j < ny; j++) {
         for (k = 0; k < nz; k++) {
            slice = ((char *)x.ptr) + k * x.pitch * ny;
            row = (int *)(slice + j * x.pitch);
            printf("d_tensor[%d][%d][%d] = %d\n", i, j, k, row[i]);
         }
      }
   }
#endif
}
__global__
void print_u16 (int nx, int ny, int nz, uint16_t* x)
{

    printf("d_tensor[%d][%d][%d] = %d\n", nx, ny, nz, x[nx + gridDim.x * blockDim.x * ny + gridDim.x * blockDim.x * gridDim.y * blockDim.y * nz]);
#if 0
   for (i = 0; i < nx; i++) {
      for (j = 0; j < ny; j++) {
         for (k = 0; k < nz; k++) {
            slice = ((char *)x.ptr) + k * x.pitch * ny;
            row = (int *)(slice + j * x.pitch);
            printf("d_tensor[%d][%d][%d] = %d\n", i, j, k, row[i]);
         }
      }
   }
#endif
}

__global__
void in_reduce(int n, int off, uint16_t *x, float *y)
{
// output idx
  int bdim = 16;
  int idxoy = blockIdx.y; 
  int idxox = blockIdx.x;
  int idxh = idxox + idxoy * gridDim.x;
  int idxo = off + idxh * n;

  for (int i = 0; i < bdim; i++) {
    for (int j = 0; j < bdim; j++) {
      int idxi  = idxh * bdim * bdim + i + j * bdim;
      y[idxo] = (float)(x[idxi]) + y[idxo];
    }
  }
#if 0
  printf("layout blockdim %d x %d, griddim %d x %d\n", blockDim.x, blockDim.y, gridDim.x, gridDim.y);
  printf("out %d x %d = %f\n", idxox, idxoy, y[idxo]);
#endif
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
void mul(int n, float* x, float* y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] * y[i];
}

__global__
void filter(int n, float* x, float* c, float* y)
{
  int idxo = blockIdx.y * blockDim.y + blockIdx.z * blockDim.z * gridDim.y * blockDim.y; 

  int idxi = idxo * n;

  for (int i = 0; i < n; i++) {
    y[idxo] = c[i] * x[idxi + i];
  }
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

