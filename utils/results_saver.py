import csv
import os
from datetime import datetime

def save_results(dense_results, sparse_results, mc2depi_results, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(out_dir, f"results_{ts}.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)

        w.writerow(["TASK 2: OPTIMIZED MATRIX MULTIPLICATION AND SPARSE MATRICES"])
        w.writerow([])

        w.writerow(["DENSE MATRIX RESULTS"])
        w.writerow(["Matrix Size", "Method", "Mean Time (s)", "Memory (MB)"])

        for r in dense_results:
            w.writerow([
                r.get("size", 0),
                r.get("name", "Unknown"),
                f"{r.get('mean', 0.0):.6f}",
                f"{r.get('memory_peak', 0.0):.2f}",
            ])

        w.writerow([])
        w.writerow(["SPARSE MATRIX RESULTS"])
        w.writerow(["Matrix Size", "Method", "Sparsity", "NNZ", "NNZ/row", "Mean Time (s)", "Memory (MB)"])

        for r in sparse_results:
            w.writerow([
                r.get("size", 0),
                r.get("label", "Unknown"),
                f"{r.get('sparsity', 0.0):.6f}",
                r.get("nnz", 0),
                f"{r.get('nnz_per_row', 0.0):.1f}",
                f"{r.get('mean', 0.0):.6f}",
                f"{r.get('memory_peak', 0.0):.2f}",
            ])

        w.writerow([])
        w.writerow(["MC2DEPI RESULTS"])
        w.writerow(["Matrix Size", "Method", "Sparsity", "NNZ", "NNZ/row", "Mean Time (s)", "Memory (MB)"])

        for r in mc2depi_results:
            w.writerow([
                r.get("size", ""),
                "mc2depi" if r.get("size") == r.get("submatrix") else f"mc2depi ({r.get('submatrix')})",
                f"{r.get('sparsity', 0.0):.6f}",
                r.get("nnz", 0),
                f"{r.get('nnz_per_row', 0.0):.1f}",
                f"{r.get('mean', 0.0):.6f}",
                f"{r.get('memory_peak', 0.0):.2f}",
            ])


    print(f"CSV saved: {csv_path}")
    return csv_path
