import numpy as np

def focal_params(pos_weight: float, gamma: float) -> dict:
    """
    pos_weight = n_class0 / n_class1
    pos_weight > 1: class-1 is minority
    pos_weight < 1: class-0 is minority
    """
    if pos_weight >= 1.0:
        imbalance    = pos_weight
        minority     = 1
        majority     = 0
        alpha_exact  = 1.0 / (1.0 + imbalance)   # weight for class-1 (minority) → >0.5... 
    else:
        imbalance    = 1.0 / pos_weight
        minority     = 0
        majority     = 1
        alpha_exact  = 1.0 / (1.0 + imbalance)   # weight for class-1 (majority) → <0.5

    gamma_safe   = round(np.clip(np.log2(imbalance) * 0.5,  0.5, 2.0), 2)
    gamma_medium = round(np.clip(np.log2(imbalance) * 0.75, 0.5, 3.0), 2)
    gamma_exact  = round(np.clip(np.log2(imbalance),        0.5, 4.0), 2)

    return {
        'alpha_exact':    round(alpha_exact, 4),
        'minority_class': minority,
        'imbalance':      round(imbalance, 2),
        'gamma_safe':     gamma_safe,
        'gamma_medium':   gamma_medium,
        'gamma_exact':    gamma_exact,
        'gamma_input':    gamma,
        'is_overcorrected': gamma > gamma_exact,
    }

print(f"{'pos_weight':>12}  {'alpha_exact':>11}  {'minority':>14}  "
      f"{'g_safe':>7}  {'g_medium':>9}  {'g_exact':>8}  {'overcorrected?':>15}")
print("-" * 90)



for pw in [0.1563]:
    r = focal_params(pw, gamma=2.0)
    print(f"{pw:>12.2f}  {r['alpha_exact']:>11.4f}  "
          f"class-{r['minority_class']} ({r['imbalance']:>5.1f}:1)  "
          f"{r['gamma_safe']:>7.2f}  {r['gamma_medium']:>9.2f}  "
          f"{r['gamma_exact']:>8.2f}  "
          f"{'YES' if r['is_overcorrected'] else 'ok':>15}")
