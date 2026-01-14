# Answers for Try This Section in 19 - SciPy Essentials

## Question 1
**Original question:** Fit an exponential decay `y = A * exp(-x/tau)` to time series data

```python
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Generate synthetic data with exponential decay
np.random.seed(42)
x_data = np.linspace(0, 10, 50)
A_true = 100
tau_true = 2.5
y_true = A_true * np.exp(-x_data / tau_true)
y_data = y_true + np.random.normal(0, 5, size=len(x_data))

# Define model
def exponential_decay(x, A, tau):
    return A * np.exp(-x / tau)

# Fit
params, covariance = curve_fit(exponential_decay, x_data, y_data, p0=[100, 2])
A_fit, tau_fit = params

print(f"True values: A={A_true}, τ={tau_true}")
print(f"Fitted values: A={A_fit:.2f}, τ={tau_fit:.2f}")

# Plot
plt.scatter(x_data, y_data, label='Data', alpha=0.5)
x_smooth = np.linspace(0, 10, 200)
y_fit = exponential_decay(x_smooth, A_fit, tau_fit)
plt.plot(x_smooth, y_fit, 'r-', label=f'Fit: τ={tau_fit:.2f}', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Intensity')
plt.legend()
plt.show()
```

## Question 2
**Original question:** Use `stats.norm.rvs()` to generate 1000 samples and plot a histogram

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Generate 1000 samples from normal distribution
samples = stats.norm.rvs(loc=200, scale=30, size=1000)

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(samples, bins=40, edgecolor='black', alpha=0.7, density=True)

# Overlay theoretical distribution
x = np.linspace(100, 300, 200)
plt.plot(x, stats.norm.pdf(x, loc=200, scale=30), 'r-', 
         linewidth=2, label='Theoretical')

plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.title('Normal Distribution: μ=200, σ=30')
plt.legend()
plt.show()

print(f"Sample mean: {np.mean(samples):.2f} (expected: 200)")
print(f"Sample std: {np.std(samples, ddof=1):.2f} (expected: 30)")
```

## Question 3
**Original question:** Interpolate between 5 data points and plot both original and interpolated

```python
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

# Original sparse data
x_original = np.array([0, 2, 4, 6, 8])
y_original = np.array([10, 25, 15, 30, 20])

# Create interpolation function
interp_func = interpolate.interp1d(x_original, y_original, kind='cubic')

# Generate smooth curve
x_smooth = np.linspace(0, 8, 100)
y_smooth = interp_func(x_smooth)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(x_original, y_original, s=100, c='red', 
            label='Original data', zorder=3)
plt.plot(x_smooth, y_smooth, 'b-', label='Cubic interpolation', linewidth=2)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Interpolation Example')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Question 4
**Original question:** Calculate the integral of sin(x) from 0 to π using `quad` (should equal 2)

```python
import numpy as np
from scipy import integrate

# Define function
def sin_func(x):
    return np.sin(x)

# Integrate from 0 to π
result, error = integrate.quad(sin_func, 0, np.pi)

print(f"Integral of sin(x) from 0 to π:")
print(f"  Result: {result:.10f}")
print(f"  Expected: 2.000000000")
print(f"  Error estimate: {error:.2e}")
print(f"  Match: {np.isclose(result, 2.0)}")

# Verify analytically: ∫sin(x)dx = -cos(x)
# From 0 to π: -cos(π) - (-cos(0)) = -(-1) - (-1) = 1 + 1 = 2
analytical = -np.cos(np.pi) - (-np.cos(0))
print(f"  Analytical result: {analytical:.10f}")
```

## Question 5
**Original question:** Create noisy data, smooth it with `savgol_filter`, and compare different window lengths

```python
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Generate noisy data
np.random.seed(42)
x = np.linspace(0, 10, 200)
y_clean = np.sin(x) + 0.5 * np.sin(3*x)
y_noisy = y_clean + np.random.normal(0, 0.2, size=len(x))

# Apply smoothing with different window lengths
windows = [5, 11, 21, 31]
plt.figure(figsize=(12, 8))

for i, window in enumerate(windows, 1):
    y_smooth = signal.savgol_filter(y_noisy, window_length=window, polyorder=3)
    
    plt.subplot(2, 2, i)
    plt.plot(x, y_noisy, 'gray', alpha=0.3, label='Noisy')
    plt.plot(x, y_clean, 'k--', label='True', linewidth=2)
    plt.plot(x, y_smooth, 'r-', label=f'Smoothed (window={window})', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Window length = {window}')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Window length effects:")
print("  Small window (5, 11): Less smoothing, preserves detail")
print("  Large window (21, 31): More smoothing, may lose features")
```
