# Hack4Dev: Transferable UnitModel for Building Age Classification

## Project Overview
The **Transferable UnitModel** is a city-agnostic machine learning architecture engineered for the **Building Age Classification Hackathon**. The objective is to classify 30m × 30m Landsat satellite pixels into **4 historical building age classes** based on four decades of multi-spectral observations (1984–2024).

The core challenge is **cross-city domain shift**: Madrid (Mediterranean; stucco/concrete) and Amsterdam (Maritime; brick/water) have drastically different baseline reflectances. Standard supervised models trained on Madrid fail in zero-shot settings on Amsterdam. 

Our solution combines **Domain-Adversarial Neural Alignment (DANN)** and **Deep Metric Learning (UnitMetricNet)** to learn city-invariant, era-sensitive representations, followed by **Prototypical Few-Shot Transfer** on Amsterdam.

---

## Deliverables & Competition Prize Alignment
Our model and evaluation protocol directly target the four competition awards:
1. **Best Low-Data F1 (25 samples/class)** (Primary Focus): Prototypical adaptation in L2-normalized embedding space guarantees high generalisation without catastrophic forgetting.
2. **Best Average F1**: Consistent, monotonic scaling across all sample volumes ($N \in [5, 25, 50, 100, 200]$).
3. **Best Peak F1**: Competitive maximum performance when all target data is leveraged.
4. **Best Originality**: Novel combination of satellite intra-annual variance, KD-Tree urban morphology context, GRL domain alignment, and Bayesian MC-Dropout uncertainty estimation.

---

## Age Classes
| Class | Madrid Period | Amsterdam Period | Historical & Spectral Signature |
|:-----:|:-------------:|:----------------:|:--------------------------------|
| **1** | Pre-1960      | Pre-1945         | Traditional masonry, settled low-variance spectral baseline |
| **2** | 1960–1984     | 1945–1984        | Pre-satellite record post-war expansion |
| **3** | 1984–2004     | 1984–2004        | Construction event visible in early Landsat observation record |
| **4** | 2004–2024     | 2004–2024        | Contemporary construction; elevated multi-spectral variance |

---

## Project Structure
The project is organized into modular Jupyter Notebooks and an end-to-end reproducible CLI script:

- **[UnitModel_1_Intro.ipynb](UnitModel_1_Intro.ipynb):** Vision, scientific methodology, class definitions, and competition deliverables.
- **[UnitModel_2_Data.ipynb](UnitModel_2_Data.ipynb):** Multi-city parquet loading, observation structure inspection, and atmospheric anomaly detection (`Blue > 15,000`).
- **[UnitModel_3_Preprocessing.ipynb](UnitModel_3_Preprocessing.ipynb):** 132-dimensional feature engineering suite:
  - Multi-observation variance (`{band}_var`)
  - Spectral indices (NDVI, NDBI, UI, MNDWI, BSI)
  - KD-Tree spatial neighborhood context (`{band}_spatial`)
  - Gap-filling and multi-era trajectory aggregation (overall, early, late, YOY)
  - Madrid-fit `StandardScaler` export to `preprocessed_data.pkl`
- **[UnitModel_4_Implementation.ipynb](UnitModel_4_Implementation.ipynb):** `UnitMetricNet` deep metric network:
  - Domain-adversarial training with Gradient Reversal Layer (GRL)
  - L2-normalized 16D hypersphere embeddings
  - Prototypical few-shot adaptation for $[5, 25, 50, 100, 200]$ samples/class
  - Bayesian MC-Dropout uncertainty quantification
  - Slide-ready summary table and F1 vs. $\log_2(N)$ error-bar plot
- **[UnitModel_5_Network_Visualization.ipynb](UnitModel_5_Network_Visualization.ipynb):** Neural network diagnostics & latent manifold visualizer:
  - Node architecture bottleneck ($132 \to 128 \to 64 \to 32 \to 16 \to 4$)
  - Forward-hook intermediate node activation heatmaps across eras
  - t-SNE / PCA latent manifold projections vs. raw input space
  - Cross-city domain alignment verification (DANN effect)
  - Gradient-based feature saliency and attribution per era
- **`unitmodel_pipeline.py`:** Standalone end-to-end Python CLI script for remote compute execution.

---

## Remote Execution Instructions
To execute on your remote compute instance:

### Environment Setup
```bash
pip install numpy pandas pyarrow scikit-learn scipy torch matplotlib
```

### Option A: End-to-End CLI Script
Run full preprocessing, base training, and few-shot evaluation:
```bash
python unitmodel_pipeline.py --mode all --device cuda --epochs 25 --few_shot_trials 10
```

Run specific stages:
```bash
# 1. Feature Preprocessing only
python unitmodel_pipeline.py --mode preprocess

# 2. Stage 1 Base Training + Stage 2 Evaluation
python unitmodel_pipeline.py --mode train --device cuda --epochs 25
```

### Option B: Interactive Jupyter Execution
Run sequentially on Jupyter Lab / Notebook:
1. `UnitModel_2_Data.ipynb`
2. `UnitModel_3_Preprocessing.ipynb` (generates `preprocessed_data.pkl`)
3. `UnitModel_4_Implementation.ipynb` (trains `UnitMetricNet` and generates transfer curve)
4. `UnitModel_5_Network_Visualization.ipynb` (generates node activation & latent manifold presentation figures)
