"""
Molecular property prediction — a self-contained QSPR pipeline (DeepChem-style).

Pipeline:  SMILES  ->  Morgan fingerprint featurization  ->  train/test split
           ->  RandomForest regression  ->  metrics + parity plot.

The target here is RDKit-computed Crippen logP. This is a *pipeline skeleton*:
to do real work, point `load_dataset()` at an experimental assay CSV
(e.g. MoleculeNet BBBP / BACE / Tox21 / ESOL) — featurization, training and
evaluation stay identical. The library of molecules is generated offline with
BRICS so the script runs with no network access.

Run:  python predict_logp.py
Deps: rdkit, scikit-learn, pandas, numpy, matplotlib
"""
from __future__ import annotations
import json, itertools, pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, BRICS, DataStructs, rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error

RDLogger.DisableLog("rdApp.*")
SEED = 42
np.random.seed(SEED)
OUT = pathlib.Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

# 15 well-known drug molecules used only as a BRICS fragment source.
SEED_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",                       # aspirin
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",                 # caffeine
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",                   # ibuprofen
    "CC(=O)Nc1ccc(O)cc1",                           # paracetamol
    "Clc1ccccc1C2=NCC(=O)Nc3ccc(cc23)Cl",          # (benzodiazepine-like)
    "COc1ccc2cc(ccc2c1)C(C)C(=O)O",                 # naproxen
    "CN1CCC[C@H]1c2cccnc2",                         # nicotine
    "OC(=O)c1ccccc1O",                              # salicylic acid
    "CC(C)NCC(O)COc1ccccc1",                        # (beta-blocker-like)
    "NC(=O)c1ccccn1",                               # nicotinamide
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",                   # theobromine/caffeine-like
    "CCO",                                          # ethanol
    "c1ccc2c(c1)cccc2",                             # naphthalene
    "CC(=O)Nc1ccc(OCC(O)CNC(C)C)cc1",              # (atenolol-like)
    "O=C(O)c1ccccc1NC(=O)c1ccccc1",                 # benzanilide acid
]


def build_library(target_n: int = 600, scan_cap: int = 6000) -> list[str]:
    """Generate a diverse, valid molecule library offline via BRICS recombination."""
    frags = set()
    for smi in SEED_SMILES:
        m = Chem.MolFromSmiles(smi)
        if m:
            frags |= set(BRICS.BRICSDecompose(m))
    frag_mols = [Chem.MolFromSmiles(f) for f in frags]
    frag_mols = [m for m in frag_mols if m is not None]

    lib: list[str] = []
    import random
    random.seed(SEED)
    builder = BRICS.BRICSBuild(frag_mols)
    for m in itertools.islice(builder, scan_cap):
        try:
            m.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(m)
            smi = Chem.MolToSmiles(m)
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            if 8 <= mol.GetNumHeavyAtoms() <= 50:    # keep drug-like sizes
                lib.append(smi)
        except Exception:
            continue
        if len(lib) >= target_n * 2:
            break
    lib = list(dict.fromkeys(lib))                    # dedup, keep order
    return lib[:target_n]


_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def featurize(smiles: list[str]):
    X = np.zeros((len(smiles), 2048), dtype=np.int8)
    for i, s in enumerate(smiles):
        fp = _GEN.GetFingerprint(Chem.MolFromSmiles(s))
        arr = np.zeros((2048,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        X[i] = arr
    return X


def load_dataset():
    smiles = build_library()
    y = np.array([Crippen.MolLogP(Chem.MolFromSmiles(s)) for s in smiles], dtype=float)
    return smiles, y


def main():
    smiles, y = load_dataset()
    X = featurize(smiles)
    print(f"library: {len(smiles)} molecules | features: {X.shape[1]} Morgan bits")

    Xtr, Xte, ytr, yte, smi_tr, smi_te = train_test_split(
        X, y, smiles, test_size=0.2, random_state=SEED
    )
    model = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=SEED)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)

    r2 = float(r2_score(yte, pred))
    rmse = float(mean_squared_error(yte, pred) ** 0.5)
    cv = cross_val_score(model, Xtr, ytr, cv=5, scoring="r2", n_jobs=-1)

    metrics = {
        "task": "regression: predict Crippen logP from Morgan fingerprints",
        "n_molecules": len(smiles),
        "n_features": int(X.shape[1]),
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "test_R2": round(r2, 4),
        "test_RMSE_logP_units": round(rmse, 4),
        "cv5_R2_mean": round(float(cv.mean()), 4),
        "cv5_R2_std": round(float(cv.std()), 4),
        "model": "RandomForestRegressor(n_estimators=300)",
        "seed": SEED,
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame({"smiles": smi_te, "logP_true": yte, "logP_pred": pred}).to_csv(
        OUT / "test_predictions.csv", index=False
    )

    # parity plot
    lo = float(min(yte.min(), pred.min())) - 0.5
    hi = float(max(yte.max(), pred.max())) + 0.5
    plt.figure(figsize=(5.2, 5.2))
    plt.scatter(yte, pred, s=16, alpha=0.6, edgecolor="none")
    plt.plot([lo, hi], [lo, hi], "k--", lw=1)
    plt.xlim(lo, hi); plt.ylim(lo, hi)
    plt.xlabel("True logP (RDKit Crippen)")
    plt.ylabel("Predicted logP (RandomForest)")
    plt.title(f"Molecular property prediction\nR2={r2:.3f}  RMSE={rmse:.3f}  (n_test={len(yte)})")
    plt.tight_layout()
    plt.savefig(OUT / "parity_plot.png", dpi=140)

    print(json.dumps(metrics, indent=2))
    print(f"\nwrote: {OUT/'metrics.json'}, {OUT/'parity_plot.png'}, {OUT/'test_predictions.csv'}")


if __name__ == "__main__":
    main()
