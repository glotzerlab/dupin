#pragma once

#include <algorithm>
#include <iostream>
#include <limits>
#include <unordered_map>
#include <vector>
#include <Eigen/Dense>


// Calculates optimal breakpoints in time-series data using memoization
class DynamicProgramming {
private:

  //stores upper triangular cost matrix efficiently
  class UpperTriangularMatrix {
  private:
    std::vector<double> matrix;
    std::vector<int> row_indices;
    int length;

    // Helper function to compute the row_indices vector
    void compute_row_indices() {
        row_indices.resize(length);
        for (int row = 0; row < length; ++row) {
            row_indices[row] = row * (2 * length - row + 1) / 2;
        }
    }

  public:
    // Constructor that initializes the matrix and row_indices
    UpperTriangularMatrix(int n) : length(n), matrix(n * (n + 1) / 2, 0.0) {
        compute_row_indices();
    }

    double &operator()(int row, int col) {
        int idx = row_indices[row] + col - row; // Use precomputed index
        return matrix[idx];
    }

    int getSize() const { return length; }
};
  // Struct for memoization key, combining start, end, and number of
  // breakpoints.
  struct MemoKey {
    int start;
    int end;
    int num_bkps;

    // Comparison operator for MemoKey.
    bool operator==(const MemoKey &other) const {
      return start == other.start && end == other.end &&
             num_bkps == other.num_bkps;
    }
  };

  // Custom XOR-bit hash function for MemoKey, avoids clustering of data in
  // unordered map to improve efficiency.
  struct MemoKeyHash {
    std::size_t operator()(const MemoKey &key) const {
      return ((std::hash<int>()(key.start) ^
               (std::hash<int>()(key.end) << 1)) >>
              1) ^
             std::hash<int>()(key.num_bkps);
    }
  };

  // Memoization map to store the cost and partition for given parameters.
  std::unordered_map<MemoKey, std::pair<double, std::vector<int>>, MemoKeyHash>
      memo;

  int num_features;    // Number of features in the dataset.
  int num_timesteps;     // Number of data points (time steps).
  int jump;              // Interval for checking potential breakpoints.
  int min_size;          // Minimum size of a segment.

  Eigen::MatrixXd data; // Matrix storing the dataset.
  UpperTriangularMatrix cost_matrix; //Matrix storing costs
  bool cost_computed = false;
  // Structure for storing linear regression parameters.
  struct linear_fit_struct {
    Eigen::MatrixXd y; // Dependent variable (labels).
    Eigen::VectorXd x; // z Independent variable (time steps).
  };
 // Scales the dataset using min-max normalization.
  void scale_data();

  // Prepares data for linear regression.
  void regression_setup(linear_fit_struct &lfit);

  // Calculates the regression line for a given data segment.
  Eigen::VectorXd regression_line(int start, int end, int dim,
                                  linear_fit_struct &lfit);

  // Generates predicted values based on the linear regression model.
  Eigen::MatrixXd predicted(int start, int end, linear_fit_struct &lfit);

  // Calculates L2 cost (Euclidean distance) between predicted and actual data.
  double l2_cost(Eigen::MatrixXd &predicted_y, int start, int end);

  // Computes the cost of a specific data segment using linear regression.
  double cost_function(int start, int end);

    // Recursive function for dynamic programming segmentation.
  std::pair<double, std::vector<int>> seg(int start, int end, int num_bkps);

// Initializes and fills the cost matrix for all data segments.
  void initialize_cost_matrix();

  // Returns the optimal set of breakpoints after segmentation.
  std::vector<int> compute_breakpoints(int num_bkps);

public:
  // Default constructor.
  DynamicProgramming();

  // Parameterized constructor.
  DynamicProgramming(const Eigen::MatrixXd &data, int jump_,
                     int min_size_);

  //Sets number of threads for parallelization
  void set_parallelization(int num_threads);

  // Calculates optimal breakpoints with given number of points.
  std::vector<int> fit(int num_bkps_in);

  // Getter functions for cost matrix.
  DynamicProgramming::UpperTriangularMatrix &getCostMatrix();
  void setCostMatrix(const DynamicProgramming::UpperTriangularMatrix &value);
};
