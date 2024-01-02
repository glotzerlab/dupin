import numpy as np
import _dupin
import random
import dupin as du
import ruptures as rpt
import time
import os
import psutil
from dupin.detect.costs import CostLinearFit


random.seed(445)
def compute_python_cost_matrix(signal: np.ndarray, min_size: int = 3):
    cost = CostLinearFit(metric = "l2")
    cost.fit(signal)
    n_samples = signal.shape[0]
    pycost_matrix = np.zeros((n_samples, n_samples))
    for start in range(n_samples):
        for end in range(start + min_size, n_samples):  
            pycost_matrix[start, end] = cost.error(start, end)  
    return pycost_matrix


def compute_cplus_cost_matrix(signal: np.ndarray):
    
    algo = _dupin.DynamicProgramming()
    algo.num_bkps = 4; 
    algo.num_timesteps = signal.shape[0]
    algo.num_parameters = signal.shape[1]
    algo.datum = np.asfortranarray(signal) 
    start1 = time.time()
    algo.initialize_cost_matrix()
    end1 = time.time()
    topcppbkps = algo.return_breakpoints()
    end2 = time.time()
    cpptime = end1-start1
    cpptime2 = end2-end1
    print("costtime: ", cpptime, "\n", "segtime: ", cpptime2)
    return topcppbkps, cpptime

def compute_cplus_cost_matrix2d(signal: np.ndarray):
    
    algo = _dupin.DupinAlgo()
    algo.num_bkps = 4; 
    algo.num_timesteps = signal.shape[0]
    algo.num_parameters = signal.shape[1]
    algo.datum = np.asfortranarray(signal) 
    start1 = time.time()
    algo.initialize_cost_matrix2d()
    
    topcppbkps = algo.getTopDownBreakpoints("old")
    end1 = time.time()
    cpptime = end1-start1
    print("cpptime2d: ", cpptime)
    return topcppbkps, cpptime




def generate_1_feature_data(size):

    data = (np.repeat([0, 200, 400, 600, 800], size / 5) + np.random.random(size)).reshape((-1, 1))
    return data

def compute_cpp_operations(data):
    topcppbkps, cpptime = compute_cplus_cost_matrix(data)
    return topcppbkps, cpptime

def compute_cpp_operations2d(data):
    topcppbkps, cpptime = compute_cplus_cost_matrix2d(data)
    return topcppbkps, cpptime


def compute_python_operations(data):
    start_time = time.time()
    dynp = rpt.Dynp(custom_cost=du.detect.CostLinearFit("l2"), jump=1)
    sweepsweep = du.detect.SweepDetector(dynp, 10)
    sweepsweep.fit(data)
    end_time = time.time()
    pythontime = end_time - start_time
    return sweepsweep.change_points_, sweepsweep.opt_change_points_, pythontime


p = psutil.Process(os.getpid())
#p.cpu_affinity([0])  # Limiting to core 0

def test_dupin():
    
    datum = generate_1_feature_data(5000)
    # C++ Operations
    total_time = 0
#    for i in range(1,11):
#        cppcost_matrix, topcppbkps, cpptime = compute_cpp_operations(current_qdata)
#        total_time = total_time + cpptime
#    print ("Average time for:",current_data.shape, ": ",  total_time/10)
    

    topcppbkps, cpptime = compute_cpp_operations(datum)
    
#    topcppbkps2, cpptime2 = compute_cpp_operations2d(datum)
    print("Top Down C++ breakpoints:", topcppbkps)
  #  print("Top Down C++ breakpoints2d:", topcppbkps2)

    # Python Operations
#    python_bkps, opt_python_bkps, pythontime = compute_python_operations(datum)

    # Comparing and printing results
#    multiplier = pythontime / cpptime
 #   print("C++ is", multiplier, "times faster!")
 #   print(f"Python breakpoints: {python_bkps} opt change points: {opt_python_bkps}")
    
    
if __name__ == "__main__":
    test_dupin()
