---
title: Homework 3
authors: Estee Rebibo (949968879) and Eden Moran (209185107) 
kernelspec:
  name: python3
  display_name: 'Python 3'
---
## Question 1:  
(a) 
```{code-cell}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N = 300  # number of training samples
sigma = np.sqrt(0.05)  # noise standard deviation

rng = np.random.default_rng(7)
x = rng.uniform(0.0, 1.0, size=N)

# Ground-truth parameters
R_max = 1.0
C50 = 0.25
n_exp = 2.0

def naka_rushton(x, R_max=R_max, C50=C50, n=n_exp):
    return R_max * (x**n) / (x**n + C50**n + 1e-12)

y_clean = naka_rushton(x)
noise = rng.normal(0.0, sigma, size=N)  # observation noise
y = y_clean + noise
```
(b)+(c)
```{code-cell}
def design_matrix(x, degree):
    return np.vander(x, N=degree + 1, increasing=True)

fits = {}
train_mse = {}

for deg in range(4):
    X = design_matrix(x, deg)
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ coeffs
    mse = np.mean((y - y_pred)**2)

    fits[deg] = coeffs
    train_mse[deg] = mse

# Show summary table
rows = []
for deg in range(4):
    row = {"degree": deg, "train_mse": train_mse[deg]}
    for i, c in enumerate(fits[deg]):
        row[f"c{i}"] = c
    rows.append(row)

df = pd.DataFrame(rows)
print("Training MSE and coefficients:")
print(df)

#(C)
x_plot = np.linspace(0, 1, 400)
y_true_plot = naka_rushton(x_plot)

plt.figure(figsize=(9, 6))
plt.scatter(x, y, s=18, alpha=0.6, label="Noisy training data")
plt.plot(x_plot, y_true_plot, linewidth=2, label="True function")

for deg in range(4):
    coeffs = fits[deg]
    X_plot = design_matrix(x_plot, deg)
    y_fit_plot = X_plot @ coeffs
    plt.plot(x_plot, y_fit_plot, linewidth=2, label=f"Degree {deg} fit")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Polynomial Regression Fits (degrees 0–3)")
plt.legend()
plt.grid(True)
plt.show()
```
(d)
```{code-cell}
# --- Generate test set ---
N_test = 300
rng = np.random.default_rng(123)   # different seed for test set
x_test = rng.uniform(0.0, 1.0, size=N_test)

# true noise-free values
y_test_clean = naka_rushton(x_test)

# noisy observations
noise_test = rng.normal(0.0, np.sqrt(0.05), size=N_test)
y_test = y_test_clean + noise_test

# --- Compute MSE for each polynomial model ---
test_mse = {}

for deg in range(4):
    coeffs = fits[deg]              # from part (b)
    X_test = design_matrix(x_test, deg)
    y_pred_test = X_test.dot(coeffs)
    mse = np.mean((y_test - y_pred_test)**2)
    test_mse[deg] = mse
    print(f"Degree {deg} test MSE: {mse:.6f}")
```

(e)  

