import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_RESULTS_DIR = os.path.join(BASE_DIR, "results_csv")
PLOT_RESULTS_DIR = os.path.join(BASE_DIR, "results_plots")

DENSE_SIZES = [32, 64, 128, 256, 512, 1024, 2048]
BLOCK_SIZES = [16, 32, 64, 128]
NUM_RUNS_DENSE = 3

SPARSE_SIZES = [32, 64, 128, 256, 512, 1024, 2048]
NUM_RUNS_SPARSE = 3

NUM_RUNS_MC2DEPI = 3
