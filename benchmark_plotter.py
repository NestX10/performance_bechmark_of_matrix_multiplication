import os
from io import StringIO
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from config import PLOT_RESULTS_DIR, CSV_RESULTS_DIR

base = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(base, CSV_RESULTS_DIR)

def latest_csv(folder):
    if not os.path.isdir(folder):
        return None
    csvs = [f for f in os.listdir(folder) if f.lower().endswith(".csv")]
    if not csvs:
        return None
    return os.path.join(folder, max(csvs, key=lambda f: os.path.getmtime(os.path.join(folder, f))))

def split_sections(csv_path):
    with open(csv_path, encoding="utf-8-sig") as f:
        lines = [l.rstrip("\n") for l in f]
    lines = [l for l in lines if l.strip() != ""]
    dense_start, sparse_start = None, None
    for i, l in enumerate(lines):
        low = l.lower()
        if "dense matrix results" in low:
            dense_start = i
        if "sparse matrix results" in low:
            sparse_start = i
    dense_lines, sparse_lines = [], []
    if dense_start is not None:
        header = lines[dense_start + 1]
        rows = lines[dense_start + 2 : (sparse_start if sparse_start else len(lines))]
        dense_lines = [header] + rows
    if sparse_start is not None:
        header = lines[sparse_start + 1]
        rows = lines[sparse_start + 2 :]
        sparse_lines = [header] + rows
    return dense_lines, sparse_lines

def load_dense_df(dense_lines):
    if not dense_lines:
        return None
    df = pd.read_csv(StringIO("\n".join(dense_lines)))
    df.columns = [c.strip() for c in df.columns]
    df["Matrix Size"] = pd.to_numeric(df["Matrix Size"], errors="coerce", downcast="integer")
    df["Mean Time (s)"] = pd.to_numeric(df["Mean Time (s)"], errors="coerce")
    df["Memory (MB)"] = pd.to_numeric(df["Memory (MB)"], errors="coerce")
    return df.dropna(subset=["Matrix Size"])

def load_sparse_df(sparse_lines):
    if not sparse_lines:
        return None
    df = pd.read_csv(StringIO("\n".join(sparse_lines)))
    df.columns = [c.strip() for c in df.columns]
    for col in ["Matrix Size", "Mean Time (s)", "Memory (MB)", "Sparsity", "NNZ/row"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Matrix Size"])
    df = df[~df["Method"].str.contains("mc2depi")]
    return df

def plot_dense(df_dense, out_dir, ts):
    if df_dense is None or df_dense.empty:
        return None, None
    df_dense = df_dense.sort_values("Matrix Size")
    sizes = sorted(df_dense["Matrix Size"].unique())
    methods = sorted(df_dense["Method"].unique())
    pivot_time = df_dense.pivot(index="Matrix Size", columns="Method", values="Mean Time (s)").loc[sizes]
    pivot_mem = df_dense.pivot(index="Matrix Size", columns="Method", values="Memory (MB)").loc[sizes]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#34495e", "#e67e22"]

    plt.figure(figsize=(10, 6))
    for i, m in enumerate(methods):
        plt.plot(sizes, pivot_time[m], marker="o", lw=2, color=colors[i % len(colors)], label=m)
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xticks(sizes, sizes)
    plt.xlabel("Matrix Size (N × N)")
    plt.ylabel("Average Time (s)")
    plt.title("Dense Matrix Multiplication – Execution Time")
    plt.grid(True, ls="--", alpha=0.3)
    plt.legend(frameon=False)
    p_time = os.path.join(out_dir, f"dense_time_{ts}.png")
    plt.savefig(p_time, dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    for i, m in enumerate(methods):
        plt.plot(sizes, pivot_mem[m], marker="o", lw=2, color=colors[i % len(colors)], label=m)
    plt.xscale("log", base=2)
    plt.xticks(sizes, sizes)
    plt.xlabel("Matrix Size (N × N)")
    plt.ylabel("Memory (MB)")
    plt.title("Dense Matrix Multiplication – Memory Usage")
    plt.grid(True, ls="--", alpha=0.3)
    plt.legend(frameon=False)
    p_mem = os.path.join(out_dir, f"dense_memory_{ts}.png")
    plt.savefig(p_mem, dpi=300)
    plt.close()

    return p_time, p_mem

def plot_sparse(df_sparse, out_dir, ts):
    if df_sparse is None or df_sparse.empty:
        return None, None
    df_sparse = df_sparse.sort_values("Matrix Size")
    sizes = sorted(df_sparse["Matrix Size"].unique())
    methods = sorted(df_sparse["Method"].unique())
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#1abc9c"]

    plt.figure(figsize=(10, 6))
    for i, m in enumerate(methods):
        sub = df_sparse[df_sparse["Method"] == m].sort_values("Matrix Size")
        plt.plot(sub["Matrix Size"], sub["Mean Time (s)"], marker="o", lw=2, color=colors[i % len(colors)], label=m)
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xticks(sizes, sizes)
    plt.xlabel("Matrix Size")
    plt.ylabel("Average Time (s)")
    plt.title("Sparse Matrix Multiplication – Execution Time")
    plt.grid(True, ls="--", alpha=0.3)
    plt.legend(frameon=False)
    p_time = os.path.join(out_dir, f"sparse_time_{ts}.png")
    plt.savefig(p_time, dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    for i, m in enumerate(methods):
        sub = df_sparse[df_sparse["Method"] == m].sort_values("Matrix Size")
        plt.plot(sub["Matrix Size"], sub["Memory (MB)"], marker="o", lw=2, color=colors[i % len(colors)], label=m)
    plt.xscale("log", base=2)
    plt.xticks(sizes, sizes)
    plt.xlabel("Matrix Size")
    plt.ylabel("Memory (MB)")
    plt.title("Sparse Matrix Multiplication – Memory Usage")
    plt.grid(True, ls="--", alpha=0.3)
    plt.legend(frameon=False)
    p_mem = os.path.join(out_dir, f"sparse_memory_{ts}.png")
    plt.savefig(p_mem, dpi=300)
    plt.close()

    return p_time, p_mem

def plot_dense_vs_sparse(df_dense, df_sparse, out_dir, ts):
    common = sorted(set(df_dense["Matrix Size"]) & set(df_sparse["Matrix Size"]))
    if len(common) == 0:
        print("⚠ No dense/sparse common sizes, skipping comparison plots.")
        return

    dense_mean = df_dense.groupby("Matrix Size")["Mean Time (s)"].mean()
    sparse_mean = df_sparse.groupby("Matrix Size")["Mean Time (s)"].mean()

    plt.figure(figsize=(10, 6))
    plt.plot(common, dense_mean.loc[common], marker="o", lw=2, label="Dense (avg of methods)")
    plt.plot(common, sparse_mean.loc[common], marker="s", lw=2, label="Sparse (avg of patterns)")
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xticks(common, common)
    plt.xlabel("Matrix Size")
    plt.ylabel("Average Time (s)")
    plt.title("Dense vs Sparse – Execution Time")
    plt.grid(True, ls="--", alpha=0.3)
    plt.legend(frameon=False)
    p_time = os.path.join(out_dir, f"dense_vs_sparse_time_{ts}.png")
    plt.savefig(p_time, dpi=300)
    plt.close()

    dense_mem = df_dense.groupby("Matrix Size")["Memory (MB)"].mean()
    sparse_mem = df_sparse.groupby("Matrix Size")["Memory (MB)"].mean()

    plt.figure(figsize=(10, 6))
    plt.plot(common, dense_mem.loc[common], marker="o", lw=2, label="Dense (avg)")
    plt.plot(common, sparse_mem.loc[common], marker="s", lw=2, label="Sparse (avg)")
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xticks(common, common)
    plt.xlabel("Matrix Size")
    plt.ylabel("Peak Memory (MB)")
    plt.title("Dense vs Sparse – Memory Usage")
    plt.grid(True, ls="--", alpha=0.3)
    plt.legend(frameon=False)
    p_mem = os.path.join(out_dir, f"dense_vs_sparse_memory_{ts}.png")
    plt.savefig(p_mem, dpi=300)
    plt.close()

def main():
    csv_path = latest_csv(results_dir)
    if not csv_path:
        print("No benchmark CSV found.")
        return
    dense_lines, sparse_lines = split_sections(csv_path)
    df_dense = load_dense_df(dense_lines)
    df_sparse = load_sparse_df(sparse_lines)
    out_dir = os.path.join(base, PLOT_RESULTS_DIR)
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_dense(df_dense, out_dir, ts)
    plot_sparse(df_sparse, out_dir, ts)
    plot_dense_vs_sparse(df_dense, df_sparse, out_dir, ts)
    print(f"Plots saved in {PLOT_RESULTS_DIR}.")

if __name__ == "__main__":
    main()
