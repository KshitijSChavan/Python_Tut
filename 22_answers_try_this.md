# Answers for Try This Section in 22 - Advanced Topics

## Question 1
**Original question:** Generate 10,000 samples from a normal distribution and verify mean and std match parameters

```python
import numpy as np
import matplotlib.pyplot as plt

# Parameters
mu = 200
sigma = 30
n_samples = 10000

# Generate samples
np.random.seed(42)
samples = np.random.normal(loc=mu, scale=sigma, size=n_samples)

# Calculate statistics
sample_mean = np.mean(samples)
sample_std = np.std(samples, ddof=1)

print(f"Parameters: μ={mu}, σ={sigma}")
print(f"Sample mean: {sample_mean:.2f} (error: {abs(sample_mean - mu):.2f})")
print(f"Sample std: {sample_std:.2f} (error: {abs(sample_std - sigma):.2f})")

# Verify with histogram
plt.figure(figsize=(10, 6))
plt.hist(samples, bins=50, density=True, alpha=0.7, edgecolor='black')

# Overlay theoretical distribution
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
plt.plot(x, 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-0.5*((x-mu)/sigma)**2), 
         'r-', linewidth=2, label='Theoretical')

plt.axvline(sample_mean, color='blue', linestyle='--', 
            label=f'Sample mean: {sample_mean:.1f}')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.title(f'Normal Distribution (N={n_samples})')
plt.legend()
plt.show()

# Statistical test
from scipy import stats
statistic, p_value = stats.normaltest(samples)
print(f"\nNormality test p-value: {p_value:.4f}")
if p_value > 0.05:
    print("Sample appears normally distributed (p > 0.05)")
```

## Question 2
**Original question:** Use Monte Carlo to estimate uncertainty in spectral index from flux measurements with errors

```python
import numpy as np
import matplotlib.pyplot as plt

# Measured values with uncertainties
flux1_measured = 245.7  # mJy at 144 MHz
flux1_error = 10.0
flux2_measured = 18.3   # mJy at 1400 MHz
flux2_error = 2.5

freq1 = 144
freq2 = 1400

# Monte Carlo simulation
n_iterations = 10000
spectral_indices = []

np.random.seed(42)
for i in range(n_iterations):
    # Draw random flux values from error distributions
    flux1_sample = np.random.normal(flux1_measured, flux1_error)
    flux2_sample = np.random.normal(flux2_measured, flux2_error)
    
    # Calculate spectral index
    if flux1_sample > 0 and flux2_sample > 0:
        alpha = np.log(flux1_sample / flux2_sample) / np.log(freq1 / freq2)
        spectral_indices.append(alpha)

spectral_indices = np.array(spectral_indices)

# Calculate statistics
mean_alpha = np.mean(spectral_indices)
std_alpha = np.std(spectral_indices)
median_alpha = np.median(spectral_indices)

print(f"Spectral index from Monte Carlo ({n_iterations} iterations):")
print(f"  Mean: {mean_alpha:.4f}")
print(f"  Std: {std_alpha:.4f}")
print(f"  Median: {median_alpha:.4f}")
print(f"  Result: α = {mean_alpha:.3f} ± {std_alpha:.3f}")

# Plot distribution
plt.figure(figsize=(10, 6))
plt.hist(spectral_indices, bins=50, edgecolor='black', alpha=0.7)
plt.axvline(mean_alpha, color='red', linestyle='--', 
            linewidth=2, label=f'Mean: {mean_alpha:.3f}')
plt.axvline(mean_alpha - std_alpha, color='red', linestyle=':', 
            alpha=0.5, label='±1σ')
plt.axvline(mean_alpha + std_alpha, color='red', linestyle=':', alpha=0.5)
plt.xlabel('Spectral Index α')
plt.ylabel('Count')
plt.title('Spectral Index Uncertainty from Monte Carlo')
plt.legend()
plt.show()
```

## Question 3
**Original question:** Create synthetic data with correlation, calculate both Pearson and Spearman coefficients

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Create correlated data
np.random.seed(42)
x = np.linspace(0, 10, 100)
y = 2 * x + 5 + np.random.normal(0, 3, size=100)  # Linear with noise

# Calculate correlations
r_pearson, p_pearson = stats.pearsonr(x, y)
r_spearman, p_spearman = stats.spearmanr(x, y)

print("Linear correlation:")
print(f"  Pearson r = {r_pearson:.4f}, p-value = {p_pearson:.4e}")
print(f"  Spearman ρ = {r_spearman:.4f}, p-value = {p_spearman:.4e}")

# Create non-linear correlation
x2 = np.linspace(0, 10, 100)
y2 = x2**2 + np.random.normal(0, 5, size=100)  # Quadratic

r_pearson2, p_pearson2 = stats.pearsonr(x2, y2)
r_spearman2, p_spearman2 = stats.spearmanr(x2, y2)

print("\nNon-linear (quadratic) correlation:")
print(f"  Pearson r = {r_pearson2:.4f} (weaker for non-linear)")
print(f"  Spearman ρ = {r_spearman2:.4f} (better for monotonic)")

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.scatter(x, y, alpha=0.5)
ax1.set_title(f'Linear: Pearson r={r_pearson:.3f}, Spearman ρ={r_spearman:.3f}')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid(True, alpha=0.3)

ax2.scatter(x2, y2, alpha=0.5, color='orange')
ax2.set_title(f'Quadratic: Pearson r={r_pearson2:.3f}, Spearman ρ={r_spearman2:.3f}')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Question 4
**Original question:** Fit a power law to source count data: N(>S) = A × S^α

```python
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Synthetic source count data
flux = np.array([10, 20, 50, 100, 200, 500, 1000])
cumulative_counts = np.array([1000, 450, 120, 35, 12, 3, 1])

# Model in log space
def log_power_law(log_S, log_A, alpha):
    return log_A + alpha * log_S

# Fit in log space
log_flux = np.log10(flux)
log_counts = np.log10(cumulative_counts)

params, cov = curve_fit(log_power_law, log_flux, log_counts)
log_A_fit, alpha_fit = params
A_fit = 10**log_A_fit

# Calculate uncertainties
perr = np.sqrt(np.diag(cov))
alpha_err = perr[1]

print(f"Power law fit: N(>S) = A × S^α")
print(f"  A = {A_fit:.2f}")
print(f"  α = {alpha_fit:.3f} ± {alpha_err:.3f}")

# Plot
flux_smooth = np.logspace(0.5, 3.5, 100)
counts_fit = A_fit * flux_smooth**alpha_fit

plt.figure(figsize=(10, 6))
plt.scatter(flux, cumulative_counts, s=100, label='Data', zorder=3)
plt.plot(flux_smooth, counts_fit, 'r-', linewidth=2, 
         label=f'Fit: N(>S) ∝ S$^{{{alpha_fit:.2f}}}$')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Flux (mJy)')
plt.ylabel('N(>S)')
plt.title('Source Counts')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Question 5
**Original question:** Create a 2D Gaussian "source" in an image and measure its peak position

```python
import numpy as np
import matplotlib.pyplot as plt

# Create image grid
image_size = 64
y, x = np.ogrid[:image_size, :image_size]

# Source parameters
peak_x = 35
peak_y = 28
amplitude = 1000
sigma = 4

# Create Gaussian source
gaussian = amplitude * np.exp(-((x - peak_x)**2 + (y - peak_y)**2) / (2 * sigma**2))

# Add noise
np.random.seed(42)
noise = np.random.normal(0, 50, (image_size, image_size))
image = gaussian + noise

# Find peak
peak_idx = np.unravel_index(np.argmax(image), image.shape)
measured_y, measured_x = peak_idx

print(f"True peak position: ({peak_x}, {peak_y})")
print(f"Measured peak position: ({measured_x}, {measured_y})")
print(f"Error: ({abs(measured_x - peak_x)}, {abs(measured_y - peak_y)}) pixels")

# Display
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, origin='lower', cmap='hot')
plt.colorbar(label='Intensity')
plt.plot(peak_x, peak_y, 'bx', markersize=15, markeredgewidth=3, 
         label='True position')
plt.plot(measured_x, measured_y, 'g+', markersize=15, markeredgewidth=3, 
         label='Measured position')
plt.xlabel('X pixel')
plt.ylabel('Y pixel')
plt.title('Image with Source')
plt.legend()

plt.subplot(1, 2, 2)
# Profile through peak
profile = image[measured_y, :]
plt.plot(profile)
plt.axvline(measured_x, color='red', linestyle='--', label='Peak')
plt.xlabel('X pixel')
plt.ylabel('Intensity')
plt.title('Horizontal Profile')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```
