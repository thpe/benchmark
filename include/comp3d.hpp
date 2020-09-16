#include <memory>

class Comp3d {
 public:
  Comp3d();
  ~Comp3d();
  Comp3d(const Comp3d&);
  Comp3d(Comp3d&&);
  Comp3d& operator=(const Comp3d&);
  Comp3d& operator=(Comp3d&&);

  void alloc (int nc, int nx, int ny);
  void load (uint16_t* data);
  void loadCoeff (float* data);
  void print ();
  void run ();
  void y (float* y);

 private:
  struct impl;
  std::unique_ptr<impl> d_ptr_;
};
