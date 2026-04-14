#!/usr/bin/env python3
"""
Final analysis: clustering structure of signal profiles vs optimal t_start.
1. PCA/t-SNE of per-sample signal vectors
2. Nearest-neighbor benchmark classification accuracy
3. Can we separate t_start=10 (CSQA+MMLU) vs t_start=14-16 (ARC-C+TruthfulQA)?
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, LeaveOneOut

RESULTS_DIR = Path("/root/autodl-tmp/loop_layer/experiments/results")
FIGURES_DIR = Path("/root/autodl-tmp/loop_layer/experiments/figures/input_dependent_analysis")

BENCHMARKS = ["ARC-C", "TruthfulQA", "CSQA", "MMLU-HS-Math"]
R30_OPTIMAL = {
    "ARC-C": 14, "TruthfulQA": 16, "CSQA": 10, "MMLU-HS-Math": 10,
}
SIGNALS = [
    "attn_entropy", "ffn_gate_norm", "layer_sim", "head_specialization",
    "logit_lens_KL", "attention_locality", "residual_write_norm",
    "participation_ratio", "prediction_flip_rate", "attn_sink_ratio",
    "residual_delta_l2", "contraction_ratio", "logit_lens_jsd_vel",
    "logit_lens_jsd_curv", "erank", "delta_erank", "attn_consensus",
    "logit_top1_margin",
]


def load_data():
    with open(RESULTS_DIR / "signal_raw_all_per_sample.json") as f:
        return json.load(f)


def build_feature_matrix(raw, n_layers):
    """
    Build [N_samples, N_features] matrix where features = signal values at all layers.
    Also return labels (benchmark name) and t_start values.
    """
    X, y_bench, y_tstart = [], [], []
    for bench in BENCHMARKS:
        for sample in raw[bench]:
            feat = []
            for sk in SIGNALS:
                feat.extend(sample[sk])
            X.append(feat)
            y_bench.append(bench)
            y_tstart.append(R30_OPTIMAL[bench])
    X = np.array(X, dtype=np.float64)
    np.nan_to_num(X, nan=0.0, copy=False)
    return X, y_bench, np.array(y_tstart)


def pca_visualization(X, y_bench, y_tstart):
    """PCA 2D projection colored by benchmark and by t_start."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Color by benchmark
    colors = {"ARC-C": "C0", "TruthfulQA": "C1", "CSQA": "C2", "MMLU-HS-Math": "C3"}
    for bench in BENCHMARKS:
        mask = [b == bench for b in y_bench]
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[bench],
                   s=40, alpha=0.7, label=f"{bench} (t*={R30_OPTIMAL[bench]})")
    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax1.set_title("PCA of signal profiles — colored by benchmark")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Color by t_start
    scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=y_tstart,
                          cmap="RdYlGn_r", s=40, alpha=0.7)
    fig.colorbar(scatter, ax=ax2, label="optimal t_start")
    ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax2.set_title("PCA of signal profiles — colored by optimal t_start")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pca_signal_profiles.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote pca_signal_profiles.png")
    print(f"  PCA explained variance: {pca.explained_variance_ratio_[:5]}")

    return pca, scaler


def classification_analysis(X, y_bench, y_tstart):
    """KNN classification: can we predict benchmark / t_start from signals?"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4-class benchmark classification
    y_bench_arr = np.array(y_bench)
    knn4 = KNeighborsClassifier(n_neighbors=5)
    scores_4class = cross_val_score(knn4, X_scaled, y_bench_arr, cv=10, scoring="accuracy")
    print(f"\n  4-class benchmark classification (10-fold CV):")
    print(f"    Accuracy: {scores_4class.mean():.3f} ± {scores_4class.std():.3f}")
    print(f"    Chance level: 0.250")

    # Binary: t_start=10 vs t_start>10
    y_binary = (y_tstart > 10).astype(int)
    knn2 = KNeighborsClassifier(n_neighbors=5)
    scores_binary = cross_val_score(knn2, X_scaled, y_binary, cv=10, scoring="accuracy")
    print(f"\n  Binary classification: t_start=10 vs t_start>10 (10-fold CV):")
    print(f"    Accuracy: {scores_binary.mean():.3f} ± {scores_binary.std():.3f}")
    print(f"    Chance level: 0.500")

    # 3-class: t_start=10, 14, 16
    knn3 = KNeighborsClassifier(n_neighbors=5)
    scores_3class = cross_val_score(knn3, X_scaled, y_tstart, cv=10, scoring="accuracy")
    print(f"\n  3-class t_start prediction (10-fold CV):")
    print(f"    Accuracy: {scores_3class.mean():.3f} ± {scores_3class.std():.3f}")
    print(f"    Chance level: 0.333 (if balanced) or 0.500 (majority class)")

    return scores_4class, scores_binary, scores_3class


def per_layer_window_analysis(raw, n_layers):
    """
    Test: using only signals from a WINDOW of layers, how well can we classify?
    This answers: which depth range carries the most input-dependent information?
    """
    scaler = StandardScaler()
    window_size = 5

    y_bench = []
    y_tstart = []
    for bench in BENCHMARKS:
        for _ in raw[bench]:
            y_bench.append(bench)
            y_tstart.append(R30_OPTIMAL[bench])
    y_bench = np.array(y_bench)
    y_tstart = np.array(y_tstart)

    results = {"start_layer": [], "acc_4class": [], "acc_binary": [], "acc_tstart": []}

    for start in range(0, n_layers - window_size + 1, 2):
        end = start + window_size
        X_window = []
        for bench in BENCHMARKS:
            for sample in raw[bench]:
                feat = []
                for sk in SIGNALS:
                    feat.extend(sample[sk][start:end])
                X_window.append(feat)
        X_window = np.array(X_window, dtype=np.float64)
        np.nan_to_num(X_window, nan=0.0, copy=False)
        X_sc = scaler.fit_transform(X_window)

        knn = KNeighborsClassifier(n_neighbors=5)
        s4 = cross_val_score(knn, X_sc, y_bench, cv=5, scoring="accuracy").mean()
        s_bin = cross_val_score(knn, X_sc, (y_tstart > 10).astype(int), cv=5, scoring="accuracy").mean()
        s_ts = cross_val_score(knn, X_sc, y_tstart, cv=5, scoring="accuracy").mean()

        results["start_layer"].append(start)
        results["acc_4class"].append(s4)
        results["acc_binary"].append(s_bin)
        results["acc_tstart"].append(s_ts)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(results["start_layer"], results["acc_4class"], "o-", label="4-class benchmark", linewidth=2)
    ax.plot(results["start_layer"], results["acc_binary"], "s-", label="binary (t*=10 vs >10)", linewidth=2)
    ax.plot(results["start_layer"], results["acc_tstart"], "^-", label="3-class t_start", linewidth=2)
    ax.axhline(0.25, color="gray", ls=":", label="4-class chance")
    ax.axhline(0.50, color="lightgray", ls=":", label="binary chance")
    ax.set_xlabel("Window start layer")
    ax.set_ylabel("Classification accuracy (5-fold CV)")
    ax.set_title(f"Classification accuracy using {window_size}-layer signal windows")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "window_classification_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote window_classification_accuracy.png")

    return results


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading data...", flush=True)
    raw = load_data()
    n_layers = len(raw[BENCHMARKS[0]][0][SIGNALS[0]])

    print("Building feature matrix...", flush=True)
    X, y_bench, y_tstart = build_feature_matrix(raw, n_layers)
    print(f"  Matrix shape: {X.shape} ({X.shape[0]} samples × {X.shape[1]} features)")

    print("PCA visualization...", flush=True)
    pca_visualization(X, y_bench, y_tstart)

    print("Classification analysis...", flush=True)
    classification_analysis(X, y_bench, y_tstart)

    print("\nPer-layer window analysis...", flush=True)
    window_results = per_layer_window_analysis(raw, n_layers)

    # Find best window for each task
    for task in ["acc_4class", "acc_binary", "acc_tstart"]:
        best_idx = np.argmax(window_results[task])
        best_start = window_results["start_layer"][best_idx]
        best_acc = window_results[task][best_idx]
        print(f"  {task}: best window starts at layer {best_start}, acc={best_acc:.3f}")

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
