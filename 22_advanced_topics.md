# Advanced Topics

You've learned the fundamentals. Here are some advanced techniques that become useful as your analyses grow more sophisticated: random number generation for simulations, Monte Carlo methods, correlation analysis, fitting power laws, working with images, and optimization.

## Random Number Generation

Simulations need random numbers. NumPy provides several distributions:

```python
import numpy as np

# Uniform distribution (equal probability between min and max)
uniform = np.random.uniform(low=100, high=300, size=1000)

# Normal (Gaussian) distribution
normal = np.random.normal(loc=200, scale=30, size=1000)
# loc = mean, scale = standard deviation

# Poisson distribution (for count data)
poisson = np.random.poisson(lam=50, size=1000)
# lam = expected count

print(f"Uniform mean: {np.mean(uniform):.2f}")
print(f"Normal mean: {np.mean(normal):.2f}")
print(f"Poisson mean: {np.mean(poisson):.2f}")
```

Why these specific distributions? Uniform for random sampling, normal for measurement errors, Poisson for photon counts or source counts.

Set a seed for reproducible random numbers:

```python
np.random.seed(42)  # Same "random" numbers every time
samples = np.random.normal(0, 1, 5)
print(samples)  # Always the same 5 numbers

np.random.seed(42)
samples2 = np.random.normal(0, 1, 5)
print(samples2)  # Identical to samples
```

This is crucial for debugging - you want reproducible results.

## Monte Carlo Simulations

Use random sampling to estimate uncertainties or probabilities. Example: propagating measurement errors through a calculation:

```python
import numpy as np
import matplotlib.pyplot as plt

# Measure two fluxes with uncertainties
flux1_measured = 245.7  # mJy
flux1_error = 10.0

flux2_measured = 189.3
flux2_error = 12.0

# Generate many realizations
n_samples = 10000
flux1_samples = np.random.normal(flux1_measured, flux1_error, n_samples)
flux2_samples = np.random.normal(flux2_measured, flux2_error, n_samples)

# Calculate derived quantity for each realization
ratio_samples = flux1_samples / flux2_samples

# Analyze distribution
ratio_mean = np.mean(ratio_samples)
ratio_std = np.std(ratio_samples)

print(f"Flux ratio: {ratio_mean:.3f} ± {ratio_std:.3f}")

# Visualize
plt.hist(ratio_samples, bins=50, edgecolor='black')
plt.axvline(ratio_mean, color='red', linestyle='--', label='Mean')
plt.xlabel('Flux Ratio')
plt.ylabel('Count')
plt.legend()
plt.show()
```

This gives you the full probability distribution of your derived quantity, not just a point estimate.

## Correlation Analysis

Do two quantities correlate? Use Pearson or Spearman correlation:

```python
import numpy as np
from scipy import stats

# Two properties: redshift and luminosity
redshift = np.array([0.5, 1.2, 0.8, 1.5, 0.3, 1.8, 0.9])
luminosity = np.array([1e23, 3e23, 1.5e23, 5e23, 5e22, 7e23, 2e23])

# Pearson correlation (linear relationship)
r_pearson, p_pearson = stats.pearsonr(redshift, luminosity)
print(f"Pearson r = {r_pearson:.3f}, p-value = {p_pearson:.3f}")

# Spearman correlation (monotonic relationship, more robust)
r_spearman, p_spearman = stats.spearmanr(redshift, luminosity)
print(f"Spearman ρ = {r_spearman:.3f}, p-value = {p_spearman:.3f}")
```

Pearson measures linear correlation, Spearman measures any monotonic relationship. Use Spearman when you expect a relationship but not necessarily linear (like power laws).

The p-value tells you if the correlation is statistically significant. Low p-value (< 0.05) means unlikely to occur by chance.

## Power Law Fitting

Many astronomical relationships follow power laws: luminosity functions, mass distributions, size-frequency relations. Fit them properly:

```python
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Data following power law N(>S) ∝ S^α
flux = np.array([10, 20, 50, 100, 200, 500])
cumulative_counts = np.array([1000, 450, 120, 35, 12, 3])

# Model: log(N) = log(A) + α*log(S)
def log_power_law(log_S, log_A, alpha):
    return log_A + alpha * log_S

# Fit in log space
log_flux = np.log10(flux)
log_counts = np.log10(cumulative_counts)

params, cov = curve_fit(log_power_law, log_flux, log_counts)
log_A_fit, alpha_fit = params

print(f"Power law index α = {alpha_fit:.3f}")

# Plot
flux_smooth = np.logspace(1, 3, 100)
counts_fit = 10**log_A_fit * flux_smooth**alpha_fit

plt.scatter(flux, cumulative_counts, s=100, label='Data')
plt.plot(flux_smooth, counts_fit, 'r-', label=f'Fit: α={alpha_fit:.2f}')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Flux (mJy)')
plt.ylabel('N(>S)')
plt.legend()
plt.show()
```

Always fit power laws in log space. Fitting in linear space gives incorrect weights to points.

## Working with Images

Radio and optical images are 2D arrays. Basic manipulation:

```python
import numpy as np
import matplotlib.pyplot as plt

# Create synthetic image (normally you'd load from FITS)
image = np.random.normal(100, 10, (128, 128))  # 128x128 pixels, mean=100, std=10

# Add a "source" - Gaussian
y, x = np.ogrid[:128, :128]
center_y, center_x = 64, 64
sigma = 5
gaussian = 500 * np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2*sigma**2))
image += gaussian

# Display
plt.imshow(image, origin='lower', cmap='hot')
plt.colorbar(label='Intensity')
plt.title('Simulated Radio Image')
plt.show()

# Extract profile
profile = image[64, :]  # Horizontal cut through center
plt.plot(profile)
plt.xlabel('X pixel')
plt.ylabel('Intensity')
plt.show()
```

For real work, use `astropy.io.fits` to read FITS images and `photutils` for source detection.

## Simple Optimization

Find parameter values that minimize or maximize a function:

```python
from scipy.optimize import minimize
import numpy as np

# Function to minimize: sum of squared residuals
def residuals(params, x_data, y_data):
    A, alpha = params
    y_model = A * x_data**alpha
    return np.sum((y_data - y_model)**2)

# Data
x_data = np.array([1, 2, 3, 4, 5])
y_data = np.array([2.1, 7.8, 16.5, 29.2, 47.1])  # Close to y = 2*x^2

# Minimize
initial_guess = [1.0, 1.0]
result = minimize(residuals, initial_guess, args=(x_data, y_data))

A_best, alpha_best = result.x
print(f"Best fit: A = {A_best:.3f}, α = {alpha_best:.3f}")
print(f"Success: {result.success}")
```

This is more general than `curve_fit` - you can minimize any function, not just fit models to data.

## Bootstrapping for Uncertainties

When you can't assume a distribution, use bootstrapping - resample your data with replacement:

```python
import numpy as np

def bootstrap_mean(data, n_bootstrap=1000):
    """Calculate mean and uncertainty via bootstrapping."""
    means = []
    for i in range(n_bootstrap):
        # Resample with replacement
        sample = np.random.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))
    
    # Uncertainty is std of bootstrap means
    return np.mean(means), np.std(means)

flux = np.array([245.7, 189.3, 312.5, 198.7, 267.4])
mean, uncertainty = bootstrap_mean(flux)

print(f"Mean: {mean:.2f} ± {uncertainty:.2f} mJy")
```

Bootstrapping is nonparametric - no assumptions about the underlying distribution.

## Binned Statistics

Calculate statistics in bins (useful for stacking, averaging):

```python
import numpy as np
from scipy import stats

# Random data
x = np.random.uniform(0, 10, 1000)
y = x**2 + np.random.normal(0, 5, 1000)

# Bin x and calculate mean y in each bin
bin_means, bin_edges, bin_number = stats.binned_statistic(
    x, y, statistic='mean', bins=10
)

# Bin centers for plotting
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

import matplotlib.pyplot as plt
plt.scatter(x, y, alpha=0.1, s=10)
plt.plot(bin_centers, bin_means, 'r-', linewidth=3, label='Binned mean')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
```

This reveals trends in noisy data.

## Things Worth Noting

**Random numbers aren't truly random.** They're pseudo-random - deterministic algorithms that produce sequences that pass statistical tests. For cryptography this matters, for simulations it's fine.

**Monte Carlo convergence.** More samples = better accuracy. Rule of thumb: use at least 1000 samples, preferably 10,000.

**Correlation ≠ causation.** High correlation doesn't mean one causes the other. Always consider physical mechanisms.

**Power law identification.** Just because data looks linear on a log-log plot doesn't guarantee it's a power law. Always check residuals and consider alternative models.

**Image processing is deep.** We've barely scratched the surface. Libraries like `scipy.ndimage`, `scikit-image`, and `photutils` provide sophisticated tools.

**Optimization can get stuck.** Minimization algorithms can find local minima instead of the global minimum. Try different initial guesses or use global optimization methods.

## Try This

1. Generate 10,000 samples from a normal distribution and verify mean and std match parameters
2. Use Monte Carlo to estimate uncertainty in spectral index from flux measurements with errors
3. Create synthetic data with correlation, calculate both Pearson and Spearman coefficients
4. Fit a power law to source count data: N(>S) = A × S^α
5. Create a 2D Gaussian "source" in an image and measure its peak position

## How This Is Typically Used in Astronomy

Monte Carlo for error propagation in complex calculations, correlation analysis for finding relationships between source properties, power law fits for luminosity functions and source counts, image analysis for morphology, optimization for model fitting, bootstrapping for robust uncertainty estimates.

These techniques appear in sophisticated analysis pipelines.

## Related Lessons

**Previous**: [21_best_practices.md](21_best_practices.md) - Writing maintainable code

**Builds on**: [14_numpy_basics.md](14_numpy_basics.md), [19_scipy_essentials.md](19_scipy_essentials.md)

**Next steps**: Learn `astropy` for astronomy-specific tools, `pandas` for complex data tables, `scikit-learn` for machine learning
