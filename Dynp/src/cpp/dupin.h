#pragma once

#include <iostream>
#include <vector>
#include <unordered_map>
#include <limits>
#include <algorithm>
#include <Eigen/Dense>
#include <omp.h>



// DynamicProgramming class for dynamic programming based segmentation.
class DynamicProgramming { //change name to Dynamic Programming
private:
    class upper_triangular_cost_matrix {
    private:
        std::vector<double> matrix;
        int size;

        int index(int i, int j) const {
            return i * (2 * size - i + 1) / 2 + (j - i);
        }

    public:
        upper_triangular_cost_matrix() : size(0) {}

        void initialize(int n) {
            size = n;
            matrix.resize(n * (n + 1) / 2, 0.0);
        }

        double& operator()(int i, int j) {
            return matrix[index(i, j)];
        }

        int getSize() const {
            return size;
        }
        
    };
    upper_triangular_cost_matrix cost_matrix;
    // Struct for memoization key, combining start, end, and number of breakpoints.
    struct MemoKey {
        int start;
        int end;
        int num_bkps;

        // Comparison operator for MemoKey.
        bool operator==(const MemoKey &other) const {
            return start == other.start && end == other.end && num_bkps == other.num_bkps;
        }
    };

    // Custom XOR-bit hash function for MemoKey, avoids clustering of data in unordered map to improve efficiency.
    struct MemoKeyHash { //test without hash function
        std::size_t operator()(const MemoKey& key) const {
            return ((std::hash<int>()(key.start) ^ (std::hash<int>()(key.end) << 1)) >> 1) ^ std::hash<int>()(key.num_bkps);
        }
    };

    // Memoization map to store the cost and partition for given parameters.
    std::unordered_map<MemoKey, std::pair<double, std::vector<int>>, MemoKeyHash> memo;

    int num_bkps; // Number of breakpoints to detect.
    int num_parameters; // Number of features in the dataset.
    int num_timesteps; // Number of data points (time steps).
    int jump; // Interval for checking potential breakpoints.
    int min_size; // Minimum size of a segment.
    Eigen::MatrixXd datum; // Matrix storing the dataset.

    

    // Structure for storing linear regression parameters.
    struct linear_fit_struct {
        Eigen::MatrixXd y; // Dependent variable (labels).
        Eigen::VectorXd x; //z Independent variable (time steps).
    };

public:
    // Default constructor.
    DynamicProgramming();

    // Parameterized constructor.
    DynamicProgramming(int num_bkps_, int num_parameters_, int num_timesteps_, int jump_, int min_size_);

    // Scales the dataset using min-max normalization.
    void scale_datum();

    // Reads input data from standard input - mainly for testing purposes.
    void read_input();

    // Prepares data for linear regression.
    void regression_setup(linear_fit_struct& lfit);

    // Calculates the regression line for a given data segment.
    Eigen::VectorXd regression_line(int start, int end, int dim, linear_fit_struct& lfit);

    // Generates predicted values based on the linear regression model.
    Eigen::MatrixXd predicted(int start, int end, linear_fit_struct& lfit);

    // Calculates L2 cost (Euclidean distance) between predicted and actual data.
    double l2_cost(Eigen::MatrixXd &predicted_y, int start, int end);

    // Computes the cost of a specific data segment using linear regression.
    double cost_function(int start, int end);

    // Initializes and fills the cost matrix for all data segments.
    void initialize_cost_matrix();

    // Recursive function for dynamic programming segmentation.
    std::pair<double, std::vector<int>> seg(int start, int end, int num_bkps);

    void set_parallelization(int num_threads); 
    // Returns the optimal set of breakpoints after segmentation.
    std::vector<int> return_breakpoints();

    // Getter functions for accessing private class members.
    int get_num_timesteps();
    int get_num_parameters();
    int get_num_bkps();
    Eigen::MatrixXd& getDatum();
    DynamicProgramming::upper_triangular_cost_matrix& getCostMatrix();

    // Setter functions for modifying private class members.
    void set_num_timesteps(int value);
    void set_num_parameters(int value);
    void set_num_bkps(int value);
    void setDatum(const Eigen::MatrixXd& value);
    void setCostMatrix(const DynamicProgramming::upper_triangular_cost_matrix& value);
};
