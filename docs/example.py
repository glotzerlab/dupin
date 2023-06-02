import numpy as np
import ruptures as rpt

import dupin as du

signal = np.load("signal.npy")

dynp = rpt.Dynp(custom_cost=du.detect.CostLinearFit())
detector = du.detect.SweepDetector(dynp, 8)

chps = detector.fit(signal)
