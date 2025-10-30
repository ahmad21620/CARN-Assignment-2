# SVHN (ATNN-2025) 4 Experimental Configurations  
**Model:** Fixed **VGG-13** (per competition rules)

---

https://www.kaggle.com/code/ahmadbshlawi/ahmad-bshlawi-assignment-2/edit

### Overview
This project explores four configurations on the **SVHN dataset** using the same VGG-13 backbone.  
The focus is on how data augmentation, MixUp intensity, and EMA decay affect **convergence speed**, **stability**, and **final validation accuracy**.

**Shared setup:**
- Optimizer: **SGD** (momentum = 0.9, Nesterov = True)  
- Scheduler: **Cosine annealing + warm-up**
- Regularization: Label smoothing = 0.02, MixUp/CutMix (prob per-exp)  
- Early stopping by **validation loss**
- EMA used for evaluation (SWA tested but not kept)

---

### Results Summary

| Config | Key changes | Time→60% | Best-loss (epoch → acc) | Peak acc | Stability |
|:--|:--|--:|--:|--:|:--|
| **exp1_baseline** | RandAug = 9, Mix p = 0.8 | **26 ep** | **60 → 67.84%** | 68.78% (e80) | Smooth, minimal oscillation |
| **exp2_strong_aug** | RandAug = 14 (+blur/persp), Mix p = 0.8 | **28 ep** | **54 → 66.02%** | 67.22% (e63) | Slower start, very stable plateau |
| **exp3_fast_opt** | EMA decay = 0.9975, Mix p = 0.6 | **20 ep** | **29 → 63.42%** | 65.16% (e44) | Fastest convergence, noisier val |
| **exp4_light_aug_fast_lr** | RandAug = 6, Mix p = 0.5 | **23 ep** | **36 → 63.26%** | 64.40% (e56) | Quick rise, light oscillation |

All configurations surpassed the **60%** validation threshold required by the competition.

---

### Interpretation
- **Augmentation strength:** Higher RandAug mag (+blur/perspective) improves robustness and smooths validation curves, though it slightly delays convergence.  
- **Mix probability:** Lower `mix_p` (0.5–0.6) speeds up early accuracy gains but reduces regularization, leading to more oscillation and lower final accuracy.  
- **EMA decay:** Smaller decay (0.9975) follows weights closely, faster updates, but more noise; larger decay (0.9995) yields smoother, higher final results.  
- **SWA:** Averaging weights under strong aug + cosine LR caused BN mismatch and accuracy drop, so EMA checkpoints were used.

---
