# SciPy Essentials

NumPy gives you fast arrays and basic math. **SciPy** builds on that with scientific algorithms - curve fitting, integration, interpolation, statistical tests, signal processing. Think of it as NumPy's scientific toolkit.

## Do You Have SciPy?

Check first:

```bash
python3 -c "import scipy; print(scipy.__version__)"
```

If not installed:

```bash
pip3 install scipy --break-system-packages
```

SciPy is organized into submodules. You import what you need: `scipy.stats` for statistics, `scipy.optimize` for fitting, `scipy.interpolate` for interpolation, etc.

## Curve Fitting - Finding Best-Fit Parameters

You have data and a model. What parameters make the model match your data? This is curve fitting:

```python
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Data: flux vs frequency
freq = np.array([144, 323, 608, 1400, 4850])
flux = np.array([245.7, 189.3, 156.2, 98.5, 45.2])

# Model: power law S = A * (freq/freq0)^alpha
def power_law(freq, A, alpha):
    freq0 = 144  # Reference frequency
    return A * (freq / freq0) ** alpha

# Fit the model to data
params, covariance = curve_fit(power_law, freq, flux, p0=[250, -0.7])

A_fit, alpha_fit = params
print(f"Best fit: A = {A_fit:.2f}, α = {alpha_fit:.3f}")

# Plot
freq_smooth = np.linspace(100, 5000, 100)
flux_fit = power_law(freq_smooth, A_fit, alpha_fit)

plt.scatter(freq, flux, label='Data', s=50)
plt.plot(freq_smooth, flux_fit, 'r-', label=f'Fit: α={alpha_fit:.3f}')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Flux (mJy)')
plt.legend()
plt.show()
```

The `curve_fit` function finds parameters that minimize the difference between your model and data. The `p0=[250, -0.7]` gives initial guesses - helps the algorithm converge faster.

The `covariance` matrix contains parameter uncertainties. Extract them:

```python
uncertainties = np.sqrt(np.diag(covariance))
A_err, alpha_err = uncertainties
print(f"α = {alpha_fit:.3f} ± {alpha_err:.3f}")
```

## Statistical Tests

SciPy has many statistical tests. Here's a common one - testing if two samples come from the same distribution:

```python
import numpy as np
from scipy import stats

# Two catalogs of flux measurements
catalog_A = np.array([245.7, 189.3, 312.5, 198.7, 267.4])
catalog_B = np.array([250.2, 195.8, 305.1, 203.4, 271.9])

# Kolmogorov-Smirnov test
statistic, p_value = stats.ks_2samp(catalog_A, catalog_B)

print(f"KS statistic: {statistic:.3f}")
print(f"p-value: {p_value:.3f}")

if p_value > 0.05:
    print("Samples likely from same distribution")
else:
    print("Samples likely from different distributions")
```

The p-value tells you: if the samples were actually from the same distribution, what's the probability of seeing this much difference? Low p-value (< 0.05) suggests they're different.

## Statistical Distributions

Generate random samples from various distributions:

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Normal distribution (Gaussian)
normal_samples = stats.norm.rvs(loc=200, scale=30, size=1000)
# loc = mean, scale = std dev

# Poisson distribution (counting statistics)
poisson_samples = stats.poisson.rvs(mu=50, size=1000)
# mu = expected count

plt.hist(normal_samples, bins=30, alpha=0.5, label='Normal')
plt.hist(poisson_samples, bins=30, alpha=0.5, label='Poisson')
plt.xlabel('Value')
plt.ylabel('Count')
plt.legend()
plt.show()
```

This is useful for Monte Carlo simulations, generating fake data for testing, or understanding measurement uncertainties.

## Interpolation - Filling Gaps

You have measurements at specific points but need values in between:

```python
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

# Sparse measurements
freq_measured = np.array([144, 323, 1400, 4850])
flux_measured = np.array([245.7, 189.3, 98.5, 45.2])

# Create interpolation function
interp_func = interpolate.interp1d(freq_measured, flux_measured, 
                                   kind='cubic')

# Generate smooth curve
freq_smooth = np.linspace(144, 4850, 100)
flux_interp = interp_func(freq_smooth)

plt.scatter(freq_measured, flux_measured, s=100, label='Measured', zorder=3)
plt.plot(freq_smooth, flux_interp, 'r-', label='Interpolated')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Flux (mJy)')
plt.legend()
plt.show()
```

The `kind='cubic'` uses cubic splines - smoother than linear interpolation. Options: `'linear'`, `'quadratic'`, `'cubic'`.

Warning: don't extrapolate far beyond your data. The interpolation knows nothing about the physics, so predictions outside your measured range are unreliable.

## Integration - Area Under Curves

Calculate the integral of a function or discrete data:

```python
import numpy as np
from scipy import integrate

# Integrate a function analytically
def gaussian(x):
    return np.exp(-x**2)

result, error = integrate.quad(gaussian, -np.inf, np.inf)
print(f"Integral: {result:.4f} (exact: √π ≈ 1.7725)")

# Integrate discrete data (trapezoid rule)
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([0, 1, 4, 9, 16, 25])  # x^2

area = integrate.trapezoid(y, x)
print(f"Area under curve: {area:.2f}")
```

The `quad` function is for continuous functions (you provide the function). For discrete data points, use `trapezoid` (formerly `trapz`).

In astronomy, you might integrate to get total flux from a spectrum or calculate enclosed mass from a density profile.

## Signal Processing - Smoothing Noisy Data

Real measurements have noise. Smoothing reveals underlying trends:

```python
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Noisy time series
t = np.linspace(0, 10, 100)
clean = np.sin(t)
noisy = clean + np.random.normal(0, 0.3, size=100)

# Savitzky-Golay filter (preserves features better than simple averaging)
smoothed = signal.savgol_filter(noisy, window_length=11, polyorder=3)

plt.plot(t, noisy, 'gray', alpha=0.5, label='Noisy')
plt.plot(t, clean, 'k--', label='True', linewidth=2)
plt.plot(t, smoothed, 'r-', label='Smoothed', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Signal')
plt.legend()
plt.show()
```

The `window_length` must be odd and determines how much smoothing. Larger values = more smoothing but less detail. The `polyorder` is the polynomial order used for fitting - keep it low (2 or 3).

## Peak Finding

Identify peaks in noisy data (useful for emission lines, pulsations):

```python
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Signal with peaks
x = np.linspace(0, 10, 1000)
y = np.sin(x) + 0.3 * np.sin(5*x) + np.random.normal(0, 0.1, 1000)

# Find peaks
peaks, properties = signal.find_peaks(y, height=0.5, distance=50)

plt.plot(x, y, 'b-', alpha=0.5)
plt.plot(x[peaks], y[peaks], 'ro', markersize=10, label='Peaks')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

print(f"Found {len(peaks)} peaks")
```

The `height=0.5` sets minimum peak height, `distance=50` ensures peaks are at least 50 points apart. Tune these to match your data.

## Things Worth Noting

**Curve fitting needs good initial guesses.** If `curve_fit` fails to converge, try different `p0` values. Look at your data and estimate reasonable starting points.

**Not all models can be fit.** Some functions are too complex or have too many parameters for reliable fitting. Start simple (power law, exponential) before trying complicated models.

**Interpolation is not extrapolation.** Don't trust interpolated values outside your data range. The interpolation function has no physics knowledge and will give nonsense results.

**Statistical tests have assumptions.** The KS test assumes continuous distributions. Other tests assume normality or independence. Read the documentation to understand what each test requires.

**Smoothing removes information.** Every smoothing algorithm trades noise reduction for loss of detail. Don't over-smooth - you might remove real features.

**Integration accuracy depends on sampling.** For `trapezoid`, more points = better accuracy. For `quad`, it adapts automatically but can fail for pathological functions.

## Try This

1. Fit an exponential decay `y = A * exp(-x/tau)` to time series data
2. Use `stats.norm.rvs()` to generate 1000 samples and plot a histogram
3. Interpolate between 5 data points and plot both original and interpolated
4. Calculate the integral of sin(x) from 0 to π using `quad` (should equal 2)
5. Create noisy data, smooth it with `savgol_filter`, and compare different window lengths

## How This Is Typically Used in Astronomy

Fitting spectral energy distributions to models, statistical comparisons of source populations, interpolating to fill gaps in observations, integrating luminosity functions, smoothing light curves, detecting peaks in spectra or time series.

SciPy is essential for quantitative analysis beyond simple statistics.

## Related Lessons

**Previous**: [18_astronomy_calculations.md](18_astronomy_calculations.md) - Domain-specific calculations

**Next**: [20_working_with_real_data.md](20_working_with_real_data.md) - Putting it all together

**Works with**: [14_numpy_basics.md](14_numpy_basics.md) - SciPy builds on NumPy arrays
