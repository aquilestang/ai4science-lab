"""
Single-cell RNA-seq analysis pipeline (scvi-tools-style goals, self-contained).

Demonstrates the core single-cell loop that tools like scVI automate:
    simulate counts (cell types + batch effect)
    -> normalize + log1p -> PCA embedding
    -> unsupervised clustering (KMeans) vs. true labels (ARI)
    -> simple batch correction (per-batch z-score) -> improved batch mixing
    -> supervised cell-type classifier on the embedding (label transfer)

Counts are simulated offline (Poisson with cell-type marker programs + a
per-batch gene-scaling effect) so it runs with no network/data download.
Swap `simulate_counts()` for a real AnnData (scanpy/scvi-tools) and the
downstream steps are identical.

Run:  python sc_pipeline.py
Deps: numpy, scikit-learn, matplotlib
"""
from __future__ import annotations
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score, accuracy_score, f1_score, silhouette_score

SEED = 0
OUT = pathlib.Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(SEED)


def simulate_counts(n_per=250, K=6, G=800, n_batches=2, n_markers=50):
    """Poisson scRNA-seq with K cell types and a per-batch gene-scaling effect."""
    base = rng.gamma(1.5, 1.0, size=G) + 0.05
    type_means = []
    for _ in range(K):
        m = base.copy()
        idx = rng.choice(G, size=n_markers, replace=False)
        m[idx] *= rng.uniform(2.5, 4.0, size=n_markers)  # moderate markers
        type_means.append(m)

    X, y_type, y_batch = [], [], []
    for b in range(n_batches):
        batch_scale = rng.uniform(0.5, 1.8, size=G)       # batch-specific gene scaling
        depth = rng.uniform(0.8, 1.2)                     # batch sequencing depth
        for k in range(K):
            lam = type_means[k] * batch_scale * depth
            counts = rng.poisson(lam, size=(n_per, G)).astype(float)
            X.append(counts)
            y_type += [k] * n_per
            y_batch += [b] * n_per
    X = np.vstack(X)
    # technical dropout: zero out ~50% of entries (scRNA-seq sparsity)
    X = X * (rng.random(X.shape) >= 0.3)
    return X, np.array(y_type), np.array(y_batch)


def normalize_log(counts):
    lib = counts.sum(axis=1, keepdims=True)
    med = np.median(lib)
    return np.log1p(counts / lib * med)


def zscore_per_batch(M, batch):
    """Toy batch correction: standardize each gene within each batch."""
    out = M.copy()
    for b in np.unique(batch):
        m = batch == b
        mu = out[m].mean(axis=0)
        sd = out[m].std(axis=0) + 1e-8
        out[m] = (out[m] - mu) / sd
    return out


def embed(M, n=50):
    return PCA(n_components=n, random_state=SEED).fit_transform(M)


def main():
    counts, y_type, y_batch = simulate_counts()
    K = len(np.unique(y_type))
    logM = normalize_log(counts)

    # --- before batch correction ---
    emb0 = embed(logM)
    km0 = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit_predict(emb0)
    ari0 = adjusted_rand_score(y_type, km0)
    batch_sil0 = silhouette_score(emb0[:, :20], y_batch)   # high => batches separate (bad)

    # --- after simple batch correction ---
    emb1 = embed(zscore_per_batch(logM, y_batch))
    km1 = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit_predict(emb1)
    ari1 = adjusted_rand_score(y_type, km1)
    batch_sil1 = silhouette_score(emb1[:, :20], y_batch)   # closer to 0 => mixed (good)

    # --- supervised cell-type classifier (label transfer) on corrected embedding ---
    Xtr, Xte, ytr, yte = train_test_split(
        emb1, y_type, test_size=0.25, random_state=SEED, stratify=y_type
    )
    clf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED).fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = accuracy_score(yte, pred)
    f1 = f1_score(yte, pred, average="macro")

    metrics = {
        "n_cells": int(counts.shape[0]),
        "n_genes": int(counts.shape[1]),
        "n_cell_types": int(K),
        "n_batches": int(len(np.unique(y_batch))),
        "clustering_ARI_before_correction": round(float(ari0), 4),
        "clustering_ARI_after_correction": round(float(ari1), 4),
        "batch_silhouette_before": round(float(batch_sil0), 4),
        "batch_silhouette_after": round(float(batch_sil1), 4),
        "celltype_classifier_accuracy": round(float(acc), 4),
        "celltype_classifier_macroF1": round(float(f1), 4),
        "note": "batch_silhouette closer to 0 = batches better mixed; ARI closer to 1 = clusters match true cell types",
        "seed": SEED,
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # --- 2x2 PCA scatter: rows = before/after correction, cols = color by type / batch ---
    fig, ax = plt.subplots(2, 2, figsize=(9, 8))
    for j, (emb, tag) in enumerate([(emb0, "raw"), (emb1, "batch-corrected")]):
        ax[j, 0].scatter(emb[:, 0], emb[:, 1], c=y_type, cmap="tab10", s=6, alpha=0.6)
        ax[j, 0].set_title(f"{tag} — colored by CELL TYPE")
        sc = ax[j, 1].scatter(emb[:, 0], emb[:, 1], c=y_batch, cmap="coolwarm", s=6, alpha=0.6)
        ax[j, 1].set_title(f"{tag} — colored by BATCH")
        for a in ax[j]:
            a.set_xlabel("PC1"); a.set_ylabel("PC2"); a.set_xticks([]); a.set_yticks([])
    fig.suptitle(
        f"scRNA-seq pipeline | ARI {ari0:.2f}→{ari1:.2f} | batch sil {batch_sil0:.2f}→{batch_sil1:.2f} "
        f"| celltype acc {acc:.2f}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT / "sc_embedding.png", dpi=140)

    print(json.dumps(metrics, indent=2))
    print(f"\nwrote: {OUT/'metrics.json'}, {OUT/'sc_embedding.png'}")


if __name__ == "__main__":
    main()
