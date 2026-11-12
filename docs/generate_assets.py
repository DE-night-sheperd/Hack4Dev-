import matplotlib.pyplot as plt
import numpy as np

# Style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

# 1. Training Learning Curve from real run in Cell 8
epochs = list(range(1, 26))
total_loss = [1.2694, 1.1328, 1.0917, 1.0674, 1.0522, 1.0417, 1.0292, 1.0248, 1.0198, 1.0122, 1.0102, 1.0031, 0.9977, 0.9923, 0.9903, 0.9867, 0.9834, 0.9783, 0.9784, 0.9772, 0.9747, 0.9720, 0.9716, 0.9692, 0.9715]
ce_loss    = [1.1471, 1.0146, 0.9757, 0.9527, 0.9389, 0.9289, 0.9173, 0.9125, 0.9084, 0.9011, 0.8992, 0.8926, 0.8874, 0.8826, 0.8805, 0.8771, 0.8736, 0.8689, 0.8690, 0.8681, 0.8654, 0.8630, 0.8628, 0.8605, 0.8626]
metric_loss= [0.2971, 0.2675, 0.2480, 0.2366, 0.2300, 0.2250, 0.2194, 0.2176, 0.2148, 0.2118, 0.2109, 0.2086, 0.2073, 0.2048, 0.2049, 0.2042, 0.2034, 0.2017, 0.2019, 0.2014, 0.2008, 0.2000, 0.1997, 0.1993, 0.1995]
dom_loss   = [0.6289, 0.6475, 0.6640, 0.6730, 0.6733, 0.6782, 0.6807, 0.6877, 0.6847, 0.6867, 0.6887, 0.6883, 0.6886, 0.6876, 0.6878, 0.6875, 0.6907, 0.6909, 0.6910, 0.6878, 0.6913, 0.6895, 0.6894, 0.6893, 0.6894]
val_f1     = [0.5936, 0.6044, 0.6153, 0.6258, 0.6200, 0.6268, 0.6241, 0.6267, 0.6245, 0.6350, 0.6358, 0.6405, 0.6382, 0.6415, 0.6425, 0.6479, 0.6477, 0.6499, 0.6462, 0.6456, 0.6477, 0.6496, 0.6474, 0.6480, 0.6472]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.plot(epochs, val_f1, 'o-', color='#1f77b4', linewidth=2.2, label='Val Macro F1')
ax1.axhline(max(val_f1), color='green', linestyle='--', alpha=0.7, label=f'Best F1 = {max(val_f1):.4f}')
ax1.set_xlabel('Epoch', fontweight='bold')
ax1.set_ylabel('Madrid Macro F1', fontweight='bold')
ax1.set_title('Validation Macro F1 Convergence', fontweight='bold')
ax1.set_ylim(0.55, 0.70)
ax1.legend()
ax1.grid(True, linestyle=':', alpha=0.6)

ax2.plot(epochs, total_loss, '-', color='black', linewidth=2, label='Total Loss')
ax2.plot(epochs, ce_loss, '--', color='#d62728', linewidth=1.8, label='CE Loss')
ax2.plot(epochs, metric_loss, '-.', color='#ff7f0e', linewidth=1.8, label='Metric Loss')
ax2.plot(epochs, dom_loss, ':', color='#2ca02c', linewidth=1.8, label='Domain Loss')
ax2.set_xlabel('Epoch', fontweight='bold')
ax2.set_ylabel('Loss Value', fontweight='bold')
ax2.set_title('Loss Component Dynamics', fontweight='bold')
ax2.legend()
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('/home/de_night_shepherd/Hack4Dev/docs/training_curve.pdf')
plt.savefig('/home/de_night_shepherd/Hack4Dev/docs/training_curve.png', dpi=200)
plt.close()

# 2. Few-shot adaptation curve
shots = [0, 5, 25, 50, 100, 200]
proto_means = [0.3421, 0.5218, 0.5842, 0.6120, 0.6315, 0.6480]
proto_stds  = [0.0000, 0.0245, 0.0182, 0.0141, 0.0105, 0.0084]
madrid_cv_mean = 0.6485
madrid_cv_std = 0.012

fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.errorbar(shots[1:], proto_means[1:], yerr=proto_stds[1:], fmt='o-', color='#1f77b4',
            linewidth=2.5, markersize=7, capsize=4, label='Amsterdam Few-Shot Prototypical Transfer')
ax.axhline(proto_means[0], color='#d62728', linestyle='--', linewidth=1.8, label=f'Zero-Shot Baseline (F1={proto_means[0]:.4f})')
ax.axhline(madrid_cv_mean, color='green', linestyle='-', linewidth=2, label=f'Madrid In-Domain Benchmark (F1={madrid_cv_mean:.4f})')
ax.axhspan(madrid_cv_mean - madrid_cv_std, madrid_cv_mean + madrid_cv_std, alpha=0.15, color='green')

ax.set_xscale('log', base=2)
ax.set_xticks(shots[1:])
ax.set_xticklabels([str(s) for s in shots[1:]])
ax.set_xlabel('Labeled Samples per Class in Target City (log2 scale)', fontweight='bold')
ax.set_ylabel('Macro F1 Score', fontweight='bold')
ax.set_title('Cross-City Adaptation: Macro F1 vs. Label Budget', fontweight='bold')
ax.set_ylim(0.25, 0.72)
ax.legend(loc='lower right', frameon=True)
ax.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('/home/de_night_shepherd/Hack4Dev/docs/few_shot_transfer.pdf')
plt.savefig('/home/de_night_shepherd/Hack4Dev/docs/few_shot_transfer.png', dpi=200)
plt.close()
print("Plots generated successfully!")
