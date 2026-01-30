import numpy as np

def geometric_affinity(f):
    d, angle, thick, vert = f
    return (
        np.exp(-d) *
        angle *
        thick *
        np.exp(-vert)
    )
