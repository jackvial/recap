import numpy as np

q = np.array([1.2, 2.8, 2.1, 0.7])

def softmax(x, temperature=1.0):
    x = np.asarray(x, dtype=float) / temperature
    x = x - np.max(x)  # numerical stability
    exp_x = np.exp(x)
    return exp_x / exp_x.sum()

def argmax_policy(x):
    p = np.zeros_like(x, dtype=float)
    p[np.argmax(x)] = 1.0
    return p

hard = argmax_policy(q)

# As we go to zero softmax converges to argmax
for T in [1.0, 0.5, 0.1, 0.01, 0.001]:
    p = softmax(q, temperature=T)
    print(f"T={T:>6}: {p}")
    print("close to argmax?", np.allclose(p, hard, atol=1e-6))
    print()