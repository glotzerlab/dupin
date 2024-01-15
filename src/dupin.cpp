#include <iostream>
#include <iomanip>
#include <limits>
#include <unordered_map>
#include <vector>
#include <Eigen/Dense>
#include <tbb/blocked_range2d.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include "dupin.h"

using namespace std;
using namespace Eigen;

DynamicProgramming::DynamicProgramming()
    : num_bkps(1), num_parameters(0), num_timesteps(0), jump(1), min_size(3) {}

DynamicProgramming::DynamicProgramming(const Eigen::MatrixXd &data, int num_bkps_, 
                                        int jump_, int min_size_)
                                        : data(data), num_bkps(num_bkps_), 
                                        jump(jump_), min_size(min_size_) {
  num_timesteps = data.rows();
  num_parameters = data.cols();
}

void DynamicProgramming::scale_data() {
  Eigen::VectorXd min_val = data.colwise().minCoeff();
  Eigen::VectorXd max_val = data.colwise().maxCoeff();
  Eigen::VectorXd range = max_val - min_val;

  for (int j = 0; j < num_parameters; ++j) {
    if (range(j) == 0.0) {
      data.col(j).setZero();
    } else {
      data.col(j) = (data.col(j).array() - min_val(j)) / range(j);
    }
  }
}
void DynamicProgramming::regression_setup(linear_fit_struct &lfit) {
  lfit.x = Eigen::VectorXd::LinSpaced(num_timesteps, 0, num_timesteps - 1) /
           (num_timesteps - 1);
  lfit.y = data;
}

Eigen::VectorXd DynamicProgramming::regression_line(int start, int end, int dim,
                                             linear_fit_struct &lfit) {
  int n = end - start;
  Eigen::VectorXd x = lfit.x.segment(start, n);
  Eigen::VectorXd y = lfit.y.col(dim).segment(start, n);

  double x_mean = x.mean();
  double y_mean = y.mean();

  Eigen::VectorXd x_centered = x.array() - x_mean;
  Eigen::VectorXd y_centered = y.array() - y_mean;

  double slope = x_centered.dot(y_centered) / x_centered.squaredNorm();
  double intercept = y_mean - slope * x_mean;

  return x.unaryExpr(
      [slope, intercept](double xi) { return slope * xi + intercept; });
}

double DynamicProgramming::l2_cost(Eigen::MatrixXd &predicted_y, int start, int end) {
  Eigen::MatrixXd diff = predicted_y.block(start, 0, end - start, num_parameters) -
                  data.block(start, 0, end - start, num_parameters);
  return std::sqrt(diff.array().square().sum());
}

Eigen::MatrixXd DynamicProgramming::predicted(int start, int end,
                                       linear_fit_struct &lfit) {
  Eigen::MatrixXd predicted_y(num_timesteps, num_parameters);
  for (int i = 0; i < num_parameters; ++i) {
    predicted_y.block(start, i, end - start, 1) =
        regression_line(start, end, i, lfit);
  }
  return predicted_y;
}

double DynamicProgramming::cost_function(int start, int end) {
  linear_fit_struct lfit;
  regression_setup(lfit);
  Eigen::MatrixXd predicted_y = predicted(start, end, lfit);
  return l2_cost(predicted_y, start, end);
}

void DynamicProgramming::initialize_cost_matrix() {
  scale_data();
  cost_matrix.initialize(num_timesteps);
  tbb::parallel_for(tbb::blocked_range<int>(0, num_timesteps),
                    [&](const tbb::blocked_range<int> &r) {
                      for (int i = r.begin(); i < r.end(); ++i) {
                        for (int j = i + min_size; j < num_timesteps; ++j) {
                          cost_matrix(i, j) = cost_function(i, j);
                        }
                      }
                    });
}

std::pair<double, std::vector<int>> DynamicProgramming::seg(int start, int end,
                                                  int num_bkps) {
  MemoKey key = {start, end, num_bkps};
  auto it = memo.find(key);
  if (it != memo.end()) {
    return it->second;
  }
  if (num_bkps == 0) {
    return {cost_matrix(start, end), {end}};
  }

  std::pair<double, std::vector<int>> best = {std::numeric_limits<double>::infinity(), {}};

  for (int bkp = start + min_size; bkp < end; bkp++) {
    if ((bkp - start) >= min_size && (end - bkp) >= min_size) {
      auto left = seg(start, bkp, num_bkps - 1);
      auto right = seg(bkp, end, 0);
      double cost = left.first + right.first;
      if (cost < best.first) {
        best.first = cost;
        best.second = left.second;
        best.second.push_back(bkp);
        best.second.insert(best.second.end(), right.second.begin(),
                           right.second.end());
      }
    }
  }

  memo[key] = best;
  return best;
}

std::vector<int> DynamicProgramming::compute_breakpoints() {
  auto result = seg(0, num_timesteps - 1, num_bkps);
  std::vector<int> breakpoints = result.second;
  std::sort(breakpoints.begin(), breakpoints.end());
  breakpoints.erase(std::unique(breakpoints.begin(), breakpoints.end()),
                    breakpoints.end());
  return breakpoints;
}

std::vector<int> DynamicProgramming::fit(){
  initialize_cost_matrix();
  return compute_breakpoints(); 
}

void set_parallelization(int num_threads) {
  static tbb::global_control gc(tbb::global_control::max_allowed_parallelism,
                                num_threads);
}

int DynamicProgramming::get_num_timesteps() { return num_timesteps; }

int DynamicProgramming::get_num_parameters() { return num_parameters; }

int DynamicProgramming::get_num_bkps() { return num_bkps; }

Eigen::MatrixXd &DynamicProgramming::getDatum() { return data; }

DynamicProgramming::UpperTriangularMatrix &
DynamicProgramming::getCostMatrix() {
  return cost_matrix;
}

void DynamicProgramming::setDatum(const Eigen::MatrixXd &value) {
  data = value;
}

void DynamicProgramming::setCostMatrix(
    const DynamicProgramming::UpperTriangularMatrix &value) {
  cost_matrix = value;
}

int main() { return 0; }