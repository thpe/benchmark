#include <Eigen/Eigen>

template <typename T, int R, int C >
class MatMul {
public:
  MatMul ()
  {
    A_ = Eigen::Matrix< T, R, C >::Random();
    B_ = Eigen::Matrix< T, C, R >::Random();
  }

  void run ()
  {
    C_ = A_ * B_;
  }

  void add ()
  {
    C_ = A_ + B_;
  }
  Eigen::Matrix< T, R, R >& res ()
  {
     return C_;
  }
private:
  Eigen::Matrix< T, R, C > A_;
  Eigen::Matrix< T, C, R > B_;
  Eigen::Matrix< T, R, R > C_;
};
