"""Surrogate model predicting 7 VLM primitives for fast optimization.

Predicts CL_0, CL_alpha, CM_0, CM_alpha, CD0_wing, CD0_body, Cn_beta --
smooth functions of geometry.  Everything else (L/D, SM, CM, CDi, penalty,
propulsion) is reconstructed analytically via reconstruct.py.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass, field
from pathlib import Path

from .features import augment_features

# ---------------------------------------------------------------------------
# VLM primitive targets -- the only things the MLP predicts
# ---------------------------------------------------------------------------
PRIMITIVE_TARGETS = [
    "CL_0", "CL_alpha", "CM_0", "CM_alpha",
    "CD0_wing", "CD0_body", "Cn_beta",
    # v2: control surface derivatives
    "Cl_beta", "CL_de", "Cm_de",
    # v3: trimmed CL from 2nd AVL run (non-linear, includes elevon effect)
    "CL",
    # v4: geometry/constraint targets (replaces _fast_* approximations)
    "struct_mass", "internal_volume", "manufacturability_score",
    "x_cg_frac", "Vs",
]

# Backward-compatible: the original 7 targets for loading legacy models
_LEGACY_TARGETS = PRIMITIVE_TARGETS[:7]

# Physical bounds for filtering divergent VLM outputs before training.
_TARGET_CLIP = {
    "CL_0":      (-1.5, 1.5),       # AVL P1/P99: -0.24 / 1.47
    "CL_alpha":  (0.1, 20.0),       # AVL P99=18.4; keep margin for high-AR wings
    "CM_0":      (-1.0, 1.0),       # AVL P1/P99: -0.96 / 0.12
    "CM_alpha":  (-15.0, 5.0),      # AVL P1/P99: -10.0 / 0.47
    "CD0_wing":  (0.001, 0.080),    # NeuralFoil-based, unchanged
    "CD0_body":  (0.0005, 0.050),   # Analytical, unchanged
    "Cn_beta":   (-0.018, 0.028),   # 3xIQR filter: removes 734 extreme outliers (1.2%), skew 3.6→1.2
    # v2 control derivatives
    "Cl_beta":   (-2.0, 0.5),       # Roll due to sideslip (negative = stable dihedral effect)
    "CL_de":     (-0.01, 0.30),     # CL per rad elevon deflection
    "Cm_de":     (-0.10, 0.01),     # Cm per rad elevon (negative = nose-down for TE-down)
    # v3: trimmed quantities
    "CL":        (-2.0, 3.0),      # Trimmed CL (from 2nd AVL run with elevon)
    # v4: geometry/constraint targets
    "struct_mass":              (0.1, 3.0),
    "internal_volume":          (0.0001, 0.03),
    "manufacturability_score":  (0.0, 1.0),
    "x_cg_frac":                (0.2, 1.0),
    "Vs":                       (5.0, 25.0),
}


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# MLP network
# ---------------------------------------------------------------------------

class _SurrogateMLP(nn.Module):
    """Feedforward MLP with BatchNorm, SiLU, Dropout."""

    def __init__(self, n_in: int, n_out: int, layers: tuple[int, ...],
                 dropout: float = 0.05):
        super().__init__()
        dims = [n_in, *layers, n_out]
        modules = []
        for i in range(len(dims) - 1):
            modules.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                modules.append(nn.BatchNorm1d(dims[i + 1]))
                modules.append(nn.SiLU())
                if dropout > 0:
                    modules.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Ensemble surrogate -- predicts 7 VLM primitives
# ---------------------------------------------------------------------------

@dataclass
class SurrogateModel:
    """Ensemble MLP surrogate trained on VLM primitives.

    Trains multiple PyTorch MLPs with bagged subsamples.
    No penalty-weighting -- physics is continuous, train uniformly.
    """

    n_ensemble: int = 5
    _n_targets: int = 10            # 7 for legacy, 10 for v2 (with control derivatives)
    _models: list = field(default_factory=list, repr=False)
    _scaler_X: StandardScaler = field(default_factory=StandardScaler, repr=False)
    _scaler_Y: StandardScaler = field(default_factory=StandardScaler, repr=False)
    _is_trained: bool = False
    _layers_used: tuple = field(default=(0,), repr=False)
    _device: torch.device = field(default_factory=_get_device, repr=False)
    _training_history: list = field(default_factory=list, repr=False)

    def fit(self, X: np.ndarray, results: list[dict],
            min_samples: int = 30, verbose: bool = True) -> bool:
        """Train ensemble on VLM evaluation results.

        Extracts primitives from results, filters divergent samples, trains
        uniformly on all data.
        """
        def _log(msg: str) -> None:
            if verbose:
                print(msg, flush=True)

        if len(results) < min_samples:
            return False

        # --- Extract primitives from evaluation results ---
        Y_rows = []
        valid_indices = []
        for i, r in enumerate(results):
            if not r.get("success", True):
                continue
            alpha_eq_rad = np.radians(r.get("alpha_eq", 0.0))
            cl_alpha = r.get("CL_alpha", 0.0)
            cm_alpha = r.get("CM_alpha", 0.0)

            cl_0 = r.get("CL_0", None)
            if cl_0 is None:
                cl_0 = r.get("CL_required", 0.0) - cl_alpha * alpha_eq_rad
            cm_0 = r.get("CM_0", None)
            if cm_0 is None:
                cm_0 = r.get("CM", 0.0) - cm_alpha * alpha_eq_rad

            cd0_wing = r.get("CD0_wing", r.get("CD0", 0.008) * 0.6)
            cd0_body = r.get("CD0_body", r.get("CD0", 0.008) * 0.4)
            cn_beta = r.get("Cn_beta", 0.0)

            # v2 control derivatives (from regeneration script)
            # Uses "Cl_beta" regenerated with consistent s_ref/b_ref normalization.
            cl_beta = r.get("Cl_beta", None)
            cl_de = r.get("CL_de", None)
            cm_de = r.get("Cm_de", None)

            # v3: trimmed CL
            cl_trimmed = r.get("CL", None)

            # Skip samples without control derivatives if we need 10+ targets
            if self._n_targets >= 10 and (cl_de is None or cm_de is None):
                continue
            if self._n_targets >= 11 and cl_trimmed is None:
                continue

            row = [cl_0, cl_alpha, cm_0, cm_alpha, cd0_wing, cd0_body, cn_beta]
            if self._n_targets >= 10:
                row.extend([cl_beta or 0.0, cl_de, cm_de])
            if self._n_targets >= 11:
                row.append(cl_trimmed)
            if self._n_targets >= 16:
                struct_m = r.get("struct_mass", None)
                int_vol = r.get("internal_volume", None)
                manuf_s = r.get("manufacturability_score", None)
                xcg_f = r.get("x_cg_frac", None)
                vs_val = r.get("Vs", None)
                if any(v is None for v in [struct_m, int_vol, manuf_s, xcg_f, vs_val]):
                    continue
                row.extend([struct_m, int_vol, manuf_s, xcg_f, vs_val])

            Y_rows.append(row)
            valid_indices.append(i)

        if len(Y_rows) < min_samples:
            return False

        Y = np.array(Y_rows)
        X_valid = X[valid_indices] if isinstance(X, np.ndarray) else np.array(X)[valid_indices]
        _log(f"  Extracted {len(Y)} valid samples from {len(results)} results")

        # --- Filter divergent samples using physical bounds ---
        mask = np.ones(len(Y), dtype=bool)
        for j, key in enumerate(PRIMITIVE_TARGETS[:self._n_targets]):
            lo, hi = _TARGET_CLIP[key]
            col_mask = (Y[:, j] >= lo) & (Y[:, j] <= hi)
            n_rejected = (~col_mask).sum()
            if n_rejected > 0:
                _log(f"    {key}: clipped {n_rejected} samples outside [{lo}, {hi}]")
            mask &= col_mask

        X_clean = X_valid[mask]
        Y_clean = Y[mask]
        _log(f"  After clip filter: {len(X_clean)} samples ({len(Y) - len(X_clean)} rejected)")

        if len(X_clean) < min_samples:
            return False

        # --- Feature augmentation ---
        X_aug = augment_features(X_clean)
        _log(f"  Features: {X_clean.shape[1]} raw -> {X_aug.shape[1]} augmented")

        # --- Scalers ---
        self._scaler_X.fit(X_aug)
        self._scaler_Y.fit(Y_clean)
        X_scaled = self._scaler_X.transform(X_aug)
        Y_scaled = self._scaler_Y.transform(Y_clean)

        # --- Target stats after scaling ---
        if verbose:
            from scipy.stats import skew as _skew
            _log(f"\n  {'Target':<12} {'Mean':>10} {'Std':>10} {'Skew':>8} {'Min':>10} {'Max':>10}")
            _log(f"  {'-'*62}")
            for j, key in enumerate(PRIMITIVE_TARGETS[:self._n_targets]):
                col = Y_clean[:, j]
                _log(f"  {key:<12} {col.mean():>10.5f} {col.std():>10.5f} {_skew(col):>8.2f} {col.min():>10.5f} {col.max():>10.5f}")
            _log("")

        n_samples = len(X_scaled)
        n_features = X_scaled.shape[1]
        n_targets = Y_scaled.shape[1]

        # --- Adapt architecture to dataset size ---
        if n_samples >= 10_000:
            layers = (256, 128, 64)
            epochs, val_frac, wd, patience = 500, 0.10, 1e-4, 50
        elif n_samples >= 2_000:
            layers = (256, 128, 64)
            epochs, val_frac, wd, patience = 400, 0.10, 1e-4, 50
        elif n_samples >= 500:
            layers = (128, 128, 64)
            epochs, val_frac, wd, patience = 400, 0.12, 3e-4, 50
        elif n_samples >= 200:
            layers = (128, 64, 32)
            epochs, val_frac, wd, patience = 300, 0.12, 5e-4, 40
        else:
            layers = (64, 32)
            epochs, val_frac, wd, patience = 300, 0.15, 1e-3, 40

        self._layers_used = layers
        batch_size = min(1024, max(256, n_samples // 32))
        _log(f"  Architecture: MLP {'-'.join(str(l) for l in layers)}")
        _log(f"  Training: {n_samples} samples, {n_features} features -> {n_targets} targets")
        _log(f"  Hyperparams: epochs={epochs}, batch={batch_size}, wd={wd}, patience={patience}")

        self._models = []
        self._training_history = []
        rng = np.random.default_rng(42)

        for _i in range(self.n_ensemble):
            # Uniform bootstrap (no penalty weighting)
            n_bag = int(0.85 * n_samples)
            idx = rng.choice(n_samples, size=n_bag, replace=True)
            X_bag, Y_bag = X_scaled[idx], Y_scaled[idx]

            # Train/val split
            n_val = max(1, int(val_frac * n_bag))
            perm = rng.permutation(n_bag)
            train_idx, val_idx = perm[n_val:], perm[:n_val]

            X_tr = torch.tensor(X_bag[train_idx], dtype=torch.float32, device=self._device)
            Y_tr = torch.tensor(Y_bag[train_idx], dtype=torch.float32, device=self._device)
            X_va = torch.tensor(X_bag[val_idx], dtype=torch.float32, device=self._device)
            Y_va = torch.tensor(Y_bag[val_idx], dtype=torch.float32, device=self._device)

            loader = DataLoader(TensorDataset(X_tr, Y_tr),
                                batch_size=min(batch_size, len(train_idx)), shuffle=True)

            model = _SurrogateMLP(n_features, n_targets, layers).to(self._device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=wd)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.5, patience=15, min_lr=1e-6)

            best_val = float("inf")
            wait = 0
            best_state = None

            for epoch in range(epochs):
                model.train()
                for xb, yb in loader:
                    loss = nn.functional.mse_loss(model(xb), yb)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    val_loss = nn.functional.mse_loss(model(X_va), Y_va).item()
                scheduler.step(val_loss)

                if val_loss < best_val - 1e-6:
                    best_val = val_loss
                    wait = 0
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                else:
                    wait += 1
                    if wait >= patience:
                        break

            stopped_epoch = epoch + 1
            if best_state is not None:
                model.load_state_dict(best_state)
            model.eval()
            self._models.append(model)
            _log(f"  Model {_i+1}/{self.n_ensemble}: best_val_loss={best_val:.6f}, stopped at epoch {stopped_epoch}/{epochs}")

        self._is_trained = True
        _log(f"\n  Ensemble training complete ({self.n_ensemble} models)")
        return True

    def _predict_ensemble(self, X: np.ndarray) -> np.ndarray:
        """Run ensemble inference -> (n_ensemble, n_samples, 7)."""
        X_aug = augment_features(X)
        X_scaled = self._scaler_X.transform(X_aug)
        X_t = torch.tensor(X_scaled, dtype=torch.float32, device=self._device)

        preds = []
        with torch.no_grad():
            for model in self._models:
                model.eval()
                Y_scaled = model(X_t).cpu().numpy()
                Y = self._scaler_Y.inverse_transform(Y_scaled)
                preds.append(Y)
        return np.array(preds)

    @property
    def target_keys(self) -> list[str]:
        """Target keys this model predicts (7 for legacy, 10 for v2)."""
        return PRIMITIVE_TARGETS[:self._n_targets]

    def predict(self, X: np.ndarray) -> dict:
        """Predict VLM primitives (ensemble mean). Returns 7 or 10 targets."""
        if not self._is_trained:
            raise RuntimeError("Surrogate not trained yet.")

        single = X.ndim == 1
        if single:
            X = X.reshape(1, -1)

        mean = self._predict_ensemble(X).mean(axis=0)
        keys = self.target_keys
        result = {key: mean[:, i] for i, key in enumerate(keys)}

        if single:
            result = {k: float(v[0]) for k, v in result.items()}
        return result

    def predict_with_uncertainty(self, X: np.ndarray) -> tuple[dict, dict]:
        """Predict with uncertainty (ensemble std)."""
        if not self._is_trained:
            raise RuntimeError("Surrogate not trained yet.")

        single = X.ndim == 1
        if single:
            X = X.reshape(1, -1)

        preds = self._predict_ensemble(X)
        mean = preds.mean(axis=0)
        std = preds.std(axis=0)

        keys = self.target_keys
        mean_dict = {key: mean[:, i] for i, key in enumerate(keys)}
        std_dict = {key: std[:, i] for i, key in enumerate(keys)}

        if single:
            mean_dict = {k: float(v[0]) for k, v in mean_dict.items()}
            std_dict = {k: float(v[0]) for k, v in std_dict.items()}
        return mean_dict, std_dict

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def architecture_str(self) -> str:
        return "-".join(str(l) for l in self._layers_used)

    def save(self, path: str | Path) -> None:
        """Save trained surrogate to disk."""
        if not self._is_trained:
            raise RuntimeError("Cannot save untrained model.")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        import joblib
        joblib.dump(self._scaler_X, path / "scaler_X.pkl")
        joblib.dump(self._scaler_Y, path / "scaler_Y.pkl")

        for i, model in enumerate(self._models):
            torch.save(model.state_dict(), path / f"model_{i}.pt")

        import json
        meta = {
            "n_ensemble": self.n_ensemble,
            "layers": list(self._layers_used),
            "target_keys": list(self.target_keys),
            "n_features_aug": self._scaler_X.n_features_in_,
            "n_targets": self._n_targets,
        }
        (path / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str | Path,
             device: torch.device | None = None) -> "SurrogateModel":
        """Load a trained surrogate from disk."""
        import json, joblib
        path = Path(path)
        device = device or _get_device()

        meta = json.loads((path / "meta.json").read_text())
        layers = tuple(meta["layers"])
        n_features = meta["n_features_aug"]
        n_targets = len(meta["target_keys"])

        sm = cls(n_ensemble=meta["n_ensemble"])
        sm._n_targets = meta.get("n_targets", len(meta["target_keys"]))
        sm._device = device
        sm._layers_used = layers
        sm._scaler_X = joblib.load(path / "scaler_X.pkl")
        sm._scaler_Y = joblib.load(path / "scaler_Y.pkl")

        sm._models = []
        for i in range(meta["n_ensemble"]):
            net = _SurrogateMLP(n_features, n_targets, layers).to(device)
            net.load_state_dict(torch.load(
                path / f"model_{i}.pt", map_location=device, weights_only=True,
            ))
            net.eval()
            sm._models.append(net)

        sm._is_trained = True
        return sm
