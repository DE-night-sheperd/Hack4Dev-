#!/usr/bin/env python3
"""
UnitModel End-to-End Pipeline for Hack4Dev Building Age Classification
Author: PixelArch / UnitModel Team
Description:
    Implements the Transferable UnitModel (Domain-Invariant Deep Metric Network).
    - Preprocessing: Multi-observation variance, KD-Tree spatial context, indices, temporal trajectories.
    - Stage 1 Base Training: Madrid training with GRL Domain Alignment on unlabeled Amsterdam features.
    - Stage 2 Adaptation: Prototypical Few-Shot Transfer for n in [5, 25, 50, 100, 200] samples/class.
    - Bayesian Uncertainty: Monte Carlo Dropout predictive entropy.
"""

import os
import sys
import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KDTree
from sklearn.metrics import f1_score, classification_report, confusion_matrix

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ── Constants ─────────────────────────────────────────────────────────────────
BASE_BANDS    = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']
VAR_BANDS     = [f'{b}_var' for b in BASE_BANDS]
SPATIAL_BANDS = [f'{b}_spatial' for b in BASE_BANDS]
ALL_BANDS     = BASE_BANDS + VAR_BANDS + SPATIAL_BANDS
INDEX_COLS    = ['NDVI', 'NDBI', 'UI', 'MNDWI', 'BSI']
BLUE_MAX      = 15_000

CLASS_EVENTS  = {'Amsterdam': 1945, 'Madrid': 1960}
CLASS_LABELS  = {
    'Amsterdam': {1: 'Pre-1945', 2: '1945-1984', 3: '1984-2004', 4: '2004-2024'},
    'Madrid':    {1: 'Pre-1960', 2: '1960-1984', 3: '1984-2004', 4: '2004-2024'},
}
EARLY_CUTOFF  = 2003
LATE_START    = 2004
PID           = ['px_key', 'py_key']


def set_seed(seed=42):
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


# ── Step 1: Preprocessing ─────────────────────────────────────────────────────
def find_data_file(filename, search_dirs):
    for d in search_dirs:
        candidate = os.path.join(d, filename)
        if os.path.exists(candidate):
            return candidate
    return None


def assign_age_class(data, events=CLASS_EVENTS):
    out = data.copy()
    out['age_class'] = pd.NA
    for city, ev in events.items():
        mask = out['city'] == city
        if mask.any():
            bins = [-np.inf, ev, 1984, 2004, np.inf]
            out.loc[mask, 'age_class'] = pd.cut(
                out.loc[mask, 'weighted_mean_year'],
                bins=bins, labels=[1, 2, 3, 4]
            ).astype('Int64')
    return out


def get_obs_stats(data, band):
    obs = []
    for slot in [1, 2, 3]:
        bc = f'{band}_{slot}'
        qc = f'qa_valid_{slot}'
        if bc in data.columns and qc in data.columns:
            val = data[bc].where(data[qc].fillna(False))
            obs.append(val)
    if not obs:
        return np.full(len(data), np.nan), np.full(len(data), np.nan)
    obs_stack = np.stack(obs, axis=1)
    best = np.nan
    for o in obs:
        best = np.where(np.isnan(best), o, best)
    var = np.nanvar(obs_stack, axis=1)
    return best, var


def make_flat_view(data):
    keep = [c for c in ['pixel_id', 'city', 'year', 'px_key', 'py_key', 'weighted_mean_year', 'age_class'] if c in data.columns]
    out = data[keep].copy()
    for band in BASE_BANDS:
        best, var = get_obs_stats(data, band)
        out[band] = best
        out[f'{band}_var'] = var
    out = out.dropna(subset=BASE_BANDS)
    out = out[out['Blue'] <= BLUE_MAX].copy()
    return out


def period_coverage(flat):
    pixels = flat[PID].drop_duplicates()
    has_early = flat[flat['year'] <= EARLY_CUTOFF].drop_duplicates(PID).assign(has_early_data=1.0)
    has_late  = flat[flat['year'] >= LATE_START].drop_duplicates(PID).assign(has_late_data=1.0)
    return pixels.merge(has_early[PID + ['has_early_data']], on=PID, how='left') \
                 .merge(has_late[PID + ['has_late_data']], on=PID, how='left') \
                 .fillna(0.0)


def fill_time_series_gaps(flat):
    year_min = flat['year'].min()
    year_max = flat['year'].max()
    all_years = list(range(year_min, year_max + 1))
    pixels = flat[PID + ['city', 'weighted_mean_year', 'age_class']].drop_duplicates(PID)
    grid = pixels.assign(_k=1).merge(pd.DataFrame({'year': all_years, '_k': 1}), on='_k').drop('_k', axis=1)
    core_bands = BASE_BANDS + VAR_BANDS
    merged = grid.merge(flat[PID + ['year'] + core_bands], on=PID + ['year'], how='left')
    merged = merged.sort_values(PID + ['year'])
    def interp(s):
        return s.interpolate(method='linear').bfill().ffill()
    for band in core_bands:
        merged[band] = merged.groupby(PID)[band].transform(interp)
    return merged


def compute_spectral_indices(df):
    df = df.copy()
    existing_indices = [c for c in df.columns if any(c.startswith(i) for i in INDEX_COLS)]
    if existing_indices:
        df = df.drop(columns=existing_indices)
    B, G, R = df['Blue'], df['Green'], df['Red']
    N, S1, S2 = df['NIR'], df['SWIR1'], df['SWIR2']
    eps = 1e-6
    df['NDVI']  = (N - R)  / (N + R  + eps)
    df['NDBI']  = (S1 - N) / (S1 + N + eps)
    df['UI']    = (S2 - N) / (S2 + N + eps)
    df['MNDWI'] = (G - S1) / (G + S1 + eps)
    df['BSI']   = ((S1 + R) - (N + B)) / ((S1 + R) + (N + B) + eps)
    return df


def add_spatial_context(flat, radius=2):
    # Prevent duplicate-merge suffix collisions (_x, _y) if function/cell is re-executed
    cols_to_drop = [c for c in flat.columns if any(c.startswith(f"{b}_spatial") for b in BASE_BANDS)]
    if cols_to_drop:
        flat = flat.drop(columns=cols_to_drop)

    pixel_means = flat.groupby(PID)[BASE_BANDS].mean().reset_index()
    coords = pixel_means[PID].to_numpy()
    arr = pixel_means[BASE_BANDS].to_numpy()
    tree = KDTree(coords)
    ind = tree.query_radius(coords, r=radius)
    spatial_arr = np.empty_like(arr)
    for i, neighbors in enumerate(ind):
        n_idx = neighbors[neighbors != i]
        spatial_arr[i] = arr[n_idx].mean(axis=0) if len(n_idx) > 0 else arr[i]
    spatial_df = pd.DataFrame(spatial_arr, columns=SPATIAL_BANDS)
    for col in PID:
        spatial_df[col] = pixel_means[col].values
    return flat.merge(spatial_df, on=PID, how='left')


def build_pixel_features(flat):
    flat = flat.copy()
    # Self-healing: if spatial bands were suffixed (_x/_y) from earlier runs, restore them
    for b in SPATIAL_BANDS:
        if b not in flat.columns and f"{b}_x" in flat.columns:
            flat[b] = flat[f"{b}_x"]

    # Verify all expected feature columns exist
    missing = [c for c in ALL_BANDS + INDEX_COLS if c not in flat.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}. Ensure add_spatial_context and compute_spectral_indices have run.")

    s = flat[flat['age_class'].notna()].copy().sort_values(PID + ['year'])
    target_cols = ALL_BANDS + INDEX_COLS
    grp = s.groupby(PID + ['age_class'])
    overall = pd.concat([
        grp[target_cols].mean().add_suffix('_mean'),
        grp[target_cols].std().add_suffix('_std')
    ], axis=1).reset_index()
    overall['age_class'] = overall['age_class'].astype(int)

    early_grp = s[s['year'] <= EARLY_CUTOFF].groupby(PID)
    early = pd.concat([
        early_grp[ALL_BANDS].mean().add_suffix('_early_mean'),
        early_grp[ALL_BANDS].std().add_suffix('_early_std')
    ], axis=1).reset_index()

    late_grp = s[s['year'] >= LATE_START].groupby(PID)
    late = pd.concat([
        late_grp[ALL_BANDS].mean().add_suffix('_late_mean'),
        late_grp[ALL_BANDS].std().add_suffix('_late_std')
    ], axis=1).reset_index()

    for b in BASE_BANDS:
        s[f'd_{b}'] = s.groupby(PID)[b].diff()
    dcols = [f'd_{b}' for b in BASE_BANDS]
    diff_grp = s.groupby(PID)[dcols]
    diff = pd.concat([
        diff_grp.mean().add_suffix('_mean'),
        diff_grp.std().add_suffix('_std')
    ], axis=1).reset_index()

    yr = s.groupby(PID)['weighted_mean_year'].first().reset_index()
    res = overall
    for extra in [early, late, diff, yr]:
        res = res.merge(extra, on=PID, how='left')
    return res


def run_preprocessing(data_dir='.', output_dir='.'):
    search_dirs = [data_dir, '.', './data', '../data', 'Building_Age_Classification/data']
    madrid_f = find_data_file('madrid_train.parquet', search_dirs)
    amsterdam_f = find_data_file('amsterdam_data.parquet', search_dirs)

    if not madrid_f or not amsterdam_f:
        raise FileNotFoundError(f"Could not locate parquet files in {search_dirs}")

    print(f"Loading raw datasets:\n  Madrid: {madrid_f}\n  Amsterdam: {amsterdam_f}")
    df_madrid = pd.read_parquet(madrid_f)
    df_amsterdam = pd.read_parquet(amsterdam_f)

    print("Assigning age classes...")
    df_madrid = assign_age_class(df_madrid)
    df_amsterdam = assign_age_class(df_amsterdam)

    print("Flattening observations and filtering outliers...")
    flat_madrid = make_flat_view(df_madrid)
    flat_amsterdam = make_flat_view(df_amsterdam)

    indicators_madrid = period_coverage(flat_madrid)
    indicators_amsterdam = period_coverage(flat_amsterdam)

    print("Filling temporal gaps...")
    flat_madrid = fill_time_series_gaps(flat_madrid)
    flat_amsterdam = fill_time_series_gaps(flat_amsterdam)

    print("Computing spectral indices & spatial morphology...")
    flat_madrid = compute_spectral_indices(flat_madrid)
    flat_amsterdam = compute_spectral_indices(flat_amsterdam)
    flat_madrid = add_spatial_context(flat_madrid)
    flat_amsterdam = add_spatial_context(flat_amsterdam)

    print("Building multi-era feature aggregation...")
    features_madrid = build_pixel_features(flat_madrid).merge(indicators_madrid, on=PID, how='left')
    features_amsterdam = build_pixel_features(flat_amsterdam).merge(indicators_amsterdam, on=PID, how='left')

    FEAT_OVERALL = [f'{b}_mean' for b in ALL_BANDS] + [f'{b}_std' for b in ALL_BANDS] + \
                   [f'{i}_mean' for i in INDEX_COLS] + [f'{i}_std'  for i in INDEX_COLS]
    FEAT_EARLY   = [f'{b}_early_mean' for b in ALL_BANDS] + [f'{b}_early_std' for b in ALL_BANDS]
    FEAT_LATE    = [f'{b}_late_mean'  for b in ALL_BANDS] + [f'{b}_late_std'  for b in ALL_BANDS]
    FEAT_YOY     = [f'd_{b}_mean' for b in BASE_BANDS] + [f'd_{b}_std' for b in BASE_BANDS]
    FEAT_IND     = ['has_early_data', 'has_late_data']
    FEATURE_COLS = FEAT_OVERALL + FEAT_EARLY + FEAT_LATE + FEAT_YOY + FEAT_IND

    TARGET_COL = 'age_class'
    X_madrid    = features_madrid[FEATURE_COLS].to_numpy(dtype=np.float64)
    y_madrid    = features_madrid[TARGET_COL].to_numpy(dtype=int)
    X_amsterdam = features_amsterdam[FEATURE_COLS].to_numpy(dtype=np.float64)
    y_amsterdam = features_amsterdam[TARGET_COL].to_numpy(dtype=int)

    print(f"Standardizing features ({len(FEATURE_COLS)} dimensions)...")
    scaler = StandardScaler()
    X_madrid = scaler.fit_transform(X_madrid)
    X_amsterdam = scaler.transform(X_amsterdam)

    bundle = {
        'X_madrid': X_madrid,
        'y_madrid': y_madrid,
        'X_amsterdam': X_amsterdam,
        'y_amsterdam': y_amsterdam,
        'feature_names': FEATURE_COLS,
        'scaler': scaler,
        'pixel_ids_madrid': features_madrid[PID].to_numpy(),
        'pixel_ids_amsterdam': features_amsterdam[PID].to_numpy(),
        'feat_groups': {
            'overall': FEAT_OVERALL,
            'early': FEAT_EARLY,
            'late': FEAT_LATE,
            'yoy': FEAT_YOY,
            'indicators': FEAT_IND
        }
    }

    out_file = os.path.join(output_dir, 'preprocessed_data.pkl')
    os.makedirs(os.path.dirname(out_file) or '.', exist_ok=True)
    with open(out_file, 'wb') as f:
        pickle.dump(bundle, f)
    print(f"Saved preprocessed data bundle ({os.path.getsize(out_file)/(1024*1024):.2f} MB) to {out_file}")
    return bundle


# ── Step 2: PyTorch Model Architecture ─────────────────────────────────────────
if TORCH_AVAILABLE:
    class GradientReversal(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, alpha):
            ctx.alpha = alpha
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output.neg() * ctx.alpha, None

    class DomainDiscriminator(nn.Module):
        def __init__(self, in_features=16, hidden_dim=32):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
        def forward(self, x):
            return self.net(x)

    class UnitMetricNet(nn.Module):
        def __init__(self, input_dim=132, embedding_dim=16, num_classes=4, dropout=0.3):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 32),
                nn.LayerNorm(32),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(32, embedding_dim)
            )
            self.classifier = nn.Linear(embedding_dim, num_classes)
            self.domain_discriminator = DomainDiscriminator(in_features=embedding_dim)

        def forward(self, x, alpha=1.0):
            raw_embed = self.backbone(x)
            norm_embed = F.normalize(raw_embed, p=2, dim=-1)
            logits = self.classifier(norm_embed)
            rev_embed = GradientReversal.apply(norm_embed, alpha)
            domain_pred = self.domain_discriminator(rev_embed)
            return logits, norm_embed, domain_pred

        def get_embedding(self, x):
            raw_embed = self.backbone(x)
            return F.normalize(raw_embed, p=2, dim=-1)


def supervised_metric_loss(embeddings, targets, margin=0.3):
    sim_mat = torch.matmul(embeddings, embeddings.T)
    target_mat = targets.unsqueeze(1) == targets.unsqueeze(0)
    pos_mask = target_mat.fill_diagonal_(False)
    neg_mask = ~target_mat
    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return torch.tensor(0.0, device=embeddings.device)
    pos_sim = sim_mat[pos_mask]
    neg_sim = sim_mat[neg_mask]
    return F.relu(neg_sim.unsqueeze(1) - pos_sim.unsqueeze(0) + margin).mean()


# ── Step 3: Training & Few-Shot Evaluation ────────────────────────────────────
def run_training(bundle, epochs=25, batch_size=256, lr=0.001, device='cpu', output_dir='.'):
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for model training.")

    X_madrid = bundle['X_madrid']
    y_madrid = bundle['y_madrid']
    X_amsterdam = bundle['X_amsterdam']
    y_amsterdam = bundle['y_amsterdam']

    dev = torch.device(device)
    print(f"Training UnitMetricNet on {dev}...")

    class_counts = np.bincount(y_madrid)[1:]
    class_weights = torch.tensor(len(y_madrid) / (4.0 * class_counts), dtype=torch.float32).to(dev)
    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    bce_loss_fn = nn.BCELoss()

    X_mad_t = torch.tensor(X_madrid, dtype=torch.float32)
    y_mad_t = torch.tensor(y_madrid - 1, dtype=torch.long)
    X_ams_t = torch.tensor(X_amsterdam, dtype=torch.float32)

    n_val = int(0.2 * len(X_mad_t))
    indices = torch.randperm(len(X_mad_t))
    train_idx, val_idx = indices[n_val:], indices[:n_val]

    mad_train_loader = DataLoader(TensorDataset(X_mad_t[train_idx], y_mad_t[train_idx]), batch_size=batch_size, shuffle=True)
    mad_val_loader   = DataLoader(TensorDataset(X_mad_t[val_idx], y_mad_t[val_idx]), batch_size=batch_size, shuffle=False)
    ams_unlabeled_loader = DataLoader(TensorDataset(X_ams_t), batch_size=batch_size, shuffle=True)

    model = UnitMetricNet(input_dim=X_madrid.shape[1], embedding_dim=16, num_classes=4, dropout=0.3).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ams_iter = iter(ams_unlabeled_loader)
    for epoch in range(epochs):
        model.train()
        p = float(epoch) / epochs
        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0

        total_loss = 0.0
        for batch_x_mad, batch_y_mad in mad_train_loader:
            batch_x_mad, batch_y_mad = batch_x_mad.to(dev), batch_y_mad.to(dev)
            try:
                batch_x_ams = next(ams_iter)[0].to(dev)
            except StopIteration:
                ams_iter = iter(ams_unlabeled_loader)
                batch_x_ams = next(ams_iter)[0].to(dev)

            logits_mad, embed_mad, dom_pred_mad = model(batch_x_mad, alpha=alpha)
            ce_loss = ce_loss_fn(logits_mad, batch_y_mad)
            met_loss = supervised_metric_loss(embed_mad, batch_y_mad)

            _, _, dom_pred_ams = model(batch_x_ams, alpha=alpha)
            dom_loss = 0.5 * (bce_loss_fn(dom_pred_mad, torch.zeros_like(dom_pred_mad)) +
                              bce_loss_fn(dom_pred_ams, torch.ones_like(dom_pred_ams)))

            loss = ce_loss + 0.2 * met_loss + 0.1 * dom_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for val_x, val_y in mad_val_loader:
                v_logits, _, _ = model(val_x.to(dev))
                val_preds.extend(v_logits.argmax(dim=-1).cpu().numpy())
                val_targets.extend(val_y.numpy())
        val_f1 = f1_score(val_targets, val_preds, average='macro')
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:2d}/{epochs}] | Madrid Val Macro F1: {val_f1:.4f}")

    # Save model weights
    model_save_path = os.path.join(output_dir, 'unitmetricnet_model.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    return model


def evaluate_few_shot(model, bundle, shots_list=[5, 25, 50, 100, 200], n_trials=10, seed=42, device='cpu', output_dir='.'):
    dev = torch.device(device)
    model.eval()
    X_ams_t = torch.tensor(bundle['X_amsterdam'], dtype=torch.float32)
    y_amsterdam = bundle['y_amsterdam']

    # Extract all embeddings
    loader = DataLoader(TensorDataset(X_ams_t), batch_size=512, shuffle=False)
    embeds = []
    with torch.no_grad():
        for bx in loader:
            e = model.get_embedding(bx[0].to(dev))
            embeds.append(e.cpu().numpy())
    E_amsterdam = np.vstack(embeds)

    # Zero shot evaluation
    with torch.no_grad():
        ams_logits, _, _ = model(X_ams_t.to(dev))
        ams_zero_pred = ams_logits.argmax(dim=-1).cpu().numpy() + 1
    f1_zero = f1_score(y_amsterdam, ams_zero_pred, average='macro')
    print(f"Zero-Shot Transfer Macro F1: {f1_zero:.4f}")

    # Prototypical Few-Shot Transfer
    rng = np.random.default_rng(seed)
    classes = np.unique(y_amsterdam)
    results = {n: [] for n in shots_list}

    for n_shots in shots_list:
        for _ in range(n_trials):
            support_idx = []
            for c in classes:
                c_idx = np.where(y_amsterdam == c)[0]
                chosen = rng.choice(c_idx, size=min(n_shots, len(c_idx)), replace=False)
                support_idx.extend(chosen)
            support_idx = np.array(support_idx)
            query_mask = np.ones(len(y_amsterdam), dtype=bool)
            query_mask[support_idx] = False

            prototypes = np.stack([
                E_amsterdam[support_idx[y_amsterdam[support_idx] == c]].mean(axis=0) for c in classes
            ])
            prototypes = prototypes / np.linalg.norm(prototypes, axis=1, keepdims=True)

            query_embeds = E_amsterdam[query_mask]
            query_targets = y_amsterdam[query_mask]

            sims = np.dot(query_embeds, prototypes.T)
            preds = classes[sims.argmax(axis=1)]
            results[n_shots].append(f1_score(query_targets, preds, average='macro'))

    means = [np.mean(results[n]) for n in shots_list]
    stds  = [np.std(results[n])  for n in shots_list]

    table_df = pd.DataFrame({
        'Samples/Class': shots_list,
        'Total Labels': [n * 4 for n in shots_list],
        'Macro F1 (Mean)': [f"{m:.4f}" for m in means],
        'Macro F1 (Std)': [f"{s:.4f}" for s in stds]
    })
    print("\n=== PROTOTYPICAL FEW-SHOT RESULTS ===")
    print(table_df.to_string(index=False))

    # Save presentation plot
    plt.figure(figsize=(9, 5))
    plt.errorbar(shots_list, means, yerr=stds, fmt='o-', color='#1f77b4', linewidth=2.5, markersize=8, capsize=5, label='UnitMetricNet Prototypical Transfer')
    plt.axhline(f1_zero, color='gray', linestyle='--', label=f'Zero-Shot Baseline (F1={f1_zero:.4f})')
    plt.xscale('log', base=2)
    plt.xticks(shots_list, [str(n) for n in shots_list])
    plt.xlabel('Labeled Samples per Class (log2 scale)', fontsize=11, fontweight='bold')
    plt.ylabel('Amsterdam Macro F1 Score', fontsize=11, fontweight='bold')
    plt.title('Few-Shot Adaptation Performance vs. Local Annotation Volume', fontsize=13, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'few_shot_transfer_f1_curve.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Saved figure to {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="UnitModel Pipeline for Hack4Dev")
    parser.add_argument('--mode', choices=['all', 'preprocess', 'train', 'eval'], default='all', help="Execution mode")
    parser.add_argument('--data_dir', default='.', help="Directory containing parquet files")
    parser.add_argument('--output_dir', default='.', help="Directory for saved outputs")
    parser.add_argument('--epochs', type=int, default=25, help="Epochs for Stage 1 base training")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--few_shot_trials', type=int, default=10, help="Number of trials per shot size")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--device', default='cuda' if (TORCH_AVAILABLE and torch.cuda.is_available()) else 'cpu', help="Torch device")
    args = parser.parse_args()

    set_seed(args.seed)

    pkl_path = os.path.join(args.output_dir, 'preprocessed_data.pkl')
    bundle = None

    if args.mode in ['all', 'preprocess']:
        bundle = run_preprocessing(args.data_dir, args.output_dir)

    if args.mode in ['all', 'train', 'eval']:
        if bundle is None:
            if not os.path.exists(pkl_path):
                raise FileNotFoundError(f"{pkl_path} not found. Run --mode preprocess first.")
            with open(pkl_path, 'rb') as f:
                bundle = pickle.load(f)

        if args.mode in ['all', 'train']:
            model = run_training(bundle, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device, output_dir=args.output_dir)
        else:
            model = UnitMetricNet(input_dim=bundle['X_madrid'].shape[1], embedding_dim=16, num_classes=4).to(torch.device(args.device))
            model_path = os.path.join(args.output_dir, 'unitmetricnet_model.pth')
            model.load_state_dict(torch.load(model_path, map_location=args.device))

        evaluate_few_shot(model, bundle, shots_list=[5, 25, 50, 100, 200], n_trials=args.few_shot_trials, seed=args.seed, device=args.device, output_dir=args.output_dir)


if __name__ == '__main__':
    main()
