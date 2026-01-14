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

---

# Answers for Try This Section in 20 - Working with Real Data

## Question 1
**Original question:** Load a CSV file with missing values and count how many NaNs are in each column

```python
import numpy as np

# Create sample CSV with missing values
sample_data = """name,ra,dec,flux,redshift
J1225+4011,187.7,12.3,245.7,1.42
J1445+3131,221.2,,312.5,
J0958+3224,149.5,32.4,,1.15
J1543+1528,235.8,15.5,289.3,0.55"""

with open('catalog_with_missing.csv', 'w') as f:
    f.write(sample_data)

# Load with genfromtxt
data = np.genfromtxt('catalog_with_missing.csv', delimiter=',', 
                     skip_header=1, usecols=(1,2,3,4), 
                     filling_values=np.nan)

# Count NaNs in each column
column_names = ['ra', 'dec', 'flux', 'redshift']
print("Missing values (NaN) per column:")
for i, name in enumerate(column_names):
    nan_count = np.sum(np.isnan(data[:, i]))
    print(f"  {name}: {nan_count} NaN values")

# Total data quality
total_rows = data.shape[0]
complete_rows = np.sum(~np.any(np.isnan(data), axis=1))
print(f"\nTotal rows: {total_rows}")
print(f"Complete rows (no NaN): {complete_rows}")
print(f"Rows with missing data: {total_rows - complete_rows}")
```

## Question 2
**Original question:** Create synthetic data with outliers, then clean it using both sigma-clipping and IQR methods

```python
import numpy as np

# Create synthetic data with outliers
np.random.seed(42)
clean_data = np.random.normal(200, 30, 100)
outliers = np.array([500, 600, 50, 30])  # Add 4 outliers
data = np.concatenate([clean_data, outliers])

print(f"Original data: {len(data)} values")

# Method 1: Sigma clipping (3-sigma)
mean = np.mean(data)
std = np.std(data, ddof=1)
sigma_cleaned = data[np.abs(data - mean) < 3 * std]
print(f"After 3-sigma clipping: {len(sigma_cleaned)} values")
print(f"  Removed: {len(data) - len(sigma_cleaned)} outliers")

# Method 2: IQR method
q1 = np.percentile(data, 25)
q3 = np.percentile(data, 75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
iqr_cleaned = data[(data >= lower_bound) & (data <= upper_bound)]
print(f"After IQR filtering: {len(iqr_cleaned)} values")
print(f"  Removed: {len(data) - len(iqr_cleaned)} outliers")

# Compare
print(f"\nComparison:")
print(f"  Sigma method removed: {data[~np.isin(data, sigma_cleaned)]}")
print(f"  IQR method removed: {data[~np.isin(data, iqr_cleaned)]}")
```

## Question 3
**Original question:** Generate two "catalogs" with some overlapping sources and cross-match them

```python
import numpy as np

# Catalog 1
catalog1_ra = np.array([187.7, 221.2, 149.5, 235.8, 201.3])
catalog1_dec = np.array([12.3, 31.5, 32.4, 15.5, 28.7])
catalog1_names = ['S1', 'S2', 'S3', 'S4', 'S5']

# Catalog 2 (some overlap with catalog 1, some different)
catalog2_ra = np.array([187.71, 149.52, 180.5, 235.79, 210.3])
catalog2_dec = np.array([12.31, 32.39, 25.0, 15.51, 20.5])
catalog2_names = ['A', 'B', 'C', 'D', 'E']

# Cross-match (tolerance = 0.1 degrees ≈ 6 arcmin)
tolerance = 0.1
matches = []

for i, (ra1, dec1) in enumerate(zip(catalog1_ra, catalog1_dec)):
    for j, (ra2, dec2) in enumerate(zip(catalog2_ra, catalog2_dec)):
        separation = np.sqrt((ra1 - ra2)**2 + (dec1 - dec2)**2)
        if separation < tolerance:
            matches.append((i, j, separation))
            print(f"Match: {catalog1_names[i]} ↔ {catalog2_names[j]} "
                  f"(sep={separation*3600:.1f} arcsec)")

print(f"\nFound {len(matches)} matches")
print(f"Catalog 1: {len(catalog1_ra)} sources, {len(matches)} matched")
print(f"Catalog 2: {len(catalog2_ra)} sources, {len(matches)} matched")
```

## Question 4
**Original question:** Simulate a time series with gaps and identify where the gaps are

```python
import numpy as np

# Create time series with gaps
time = np.array([1, 2, 3, 4, 10, 11, 12, 20, 21, 22, 23])
flux = np.array([245, 250, 248, 252, 255, 253, 251, 260, 258, 262, 259])

# Find gaps
time_diff = np.diff(time)
median_cadence = np.median(time_diff)
threshold = 3 * median_cadence  # Gaps larger than 3x median

gap_indices = np.where(time_diff > threshold)[0]

print(f"Time series: {len(time)} observations")
print(f"Median cadence: {median_cadence:.1f} days")
print(f"Gap threshold: {threshold:.1f} days")
print(f"\nFound {len(gap_indices)} gaps:")

for idx in gap_indices:
    gap_start = time[idx]
    gap_end = time[idx + 1]
    gap_size = gap_end - gap_start
    print(f"  Gap between day {gap_start} and {gap_end} ({gap_size} days)")

# Visualize
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))
plt.plot(time, flux, 'o-')
for idx in gap_indices:
    plt.axvspan(time[idx], time[idx+1], alpha=0.2, color='red')
plt.xlabel('Time (days)')
plt.ylabel('Flux (mJy)')
plt.title('Time Series with Gaps (red regions)')
plt.grid(True, alpha=0.3)
plt.show()
```

## Question 5
**Original question:** Create a quality flag array and filter data to keep only "good" measurements

```python
import numpy as np

# Simulated catalog
sources = np.array([
    ['J1225+4011', 187.7, 12.3, 245.7, 0],  # Good
    ['J1445+3131', 221.2, 31.5, 312.5, 1],  # Marginal
    ['J0958+3224', 149.5, 32.4, 198.7, 0],  # Good
    ['J1543+1528', 235.8, 15.5, 289.3, 2],  # Bad
    ['J2134+0042', 323.5, 0.7, 156.2, 0],   # Good
], dtype=object)

names = sources[:, 0]
ra = sources[:, 1].astype(float)
dec = sources[:, 2].astype(float)
flux = sources[:, 3].astype(float)
quality = sources[:, 4].astype(int)

# Count by quality
print("Quality flag distribution:")
print(f"  Flag 0 (good): {np.sum(quality == 0)} sources")
print(f"  Flag 1 (marginal): {np.sum(quality == 1)} sources")
print(f"  Flag 2 (bad): {np.sum(quality == 2)} sources")

# Filter to good data only
good_mask = quality == 0
names_good = names[good_mask]
flux_good = flux[good_mask]

print(f"\nGood quality sources:")
for name, f in zip(names_good, flux_good):
    print(f"  {name}: {f} mJy")

print(f"\nFiltered: {len(flux_good)}/{len(flux)} sources kept")
```

---

# Answers for Try This Section in 21 - Best Practices

## Question 1
**Original question:** Take one of your earlier scripts and add descriptive variable names everywhere

```python
# BEFORE (bad variable names)
x = 245.7
y = 144
z = x * (y/1400)**-0.7

# AFTER (descriptive names)
flux_at_144MHz = 245.7  # mJy
reference_frequency = 144  # MHz
target_frequency = 1400  # MHz
spectral_index = -0.7
flux_at_1400MHz = flux_at_144MHz * (reference_frequency / target_frequency) ** spectral_index

print(f"Flux at {reference_frequency} MHz: {flux_at_144MHz} mJy")
print(f"Predicted flux at {target_frequency} MHz: {flux_at_1400MHz:.2f} mJy")
```

## Question 2
**Original question:** Write docstrings for 3 functions you've created

```python
def calculate_mean(values):
    """
    Calculate the arithmetic mean of a list of numbers.
    
    Parameters:
        values (list): List of numerical values
        
    Returns:
        float: Mean value, or None if list is empty
        
    Example:
        >>> flux = [245.7, 189.3, 312.5]
        >>> mean = calculate_mean(flux)
        >>> print(mean)
        249.17
    """
    if len(values) == 0:
        return None
    return sum(values) / len(values)


def is_detected(flux, noise, sigma=5):
    """
    Check if a source meets detection threshold.
    
    Uses N-sigma detection criterion where source flux
    must exceed N times the noise level.
    
    Parameters:
        flux (float): Measured flux in mJy
        noise (float): Noise level in mJy
        sigma (int): Detection threshold in sigma (default: 5)
        
    Returns:
        bool: True if flux > sigma * noise
        
    Example:
        >>> is_detected(150.0, 10.0, sigma=5)
        True
        >>> is_detected(40.0, 10.0, sigma=5)
        False
    """
    return flux > sigma * noise


def extrapolate_flux(flux_ref, freq_ref, freq_target, alpha):
    """
    Extrapolate flux to a different frequency using power law.
    
    Uses the relation S ∝ ν^α to predict flux at target frequency
    given a measurement at reference frequency.
    
    Parameters:
        flux_ref (float): Flux at reference frequency (mJy)
        freq_ref (float): Reference frequency (MHz)
        freq_target (float): Target frequency (MHz)
        alpha (float): Spectral index
        
    Returns:
        float: Predicted flux at target frequency (mJy)
        
    Notes:
        - Steep spectrum: α < -0.5
        - Flat spectrum: -0.5 < α < 0.5
        - Inverted spectrum: α > 0.5
        
    Example:
        >>> flux = extrapolate_flux(245.7, 144, 1400, -0.7)
        >>> print(f"Flux: {flux:.2f} mJy")
        Flux: 35.57 mJy
    """
    return flux_ref * (freq_target / freq_ref) ** alpha
```

## Question 3
**Original question:** Add error handling to a file-loading function (check if file exists, handle parse errors)

```python
import os
import numpy as np

def load_catalog_safe(filename):
    """
    Load catalog with comprehensive error handling.
    
    Parameters:
        filename (str): Path to catalog file
        
    Returns:
        numpy.ndarray: Loaded data, or None if loading fails
    """
    # Check if file exists
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found")
        print(f"Current directory: {os.getcwd()}")
        return None
    
    # Check if file is readable
    if not os.access(filename, os.R_OK):
        print(f"Error: No permission to read '{filename}'")
        return None
    
    # Try to load
    try:
        data = np.loadtxt(filename)
        
        # Validate data
        if len(data) == 0:
            print(f"Warning: File '{filename}' is empty")
            return None
            
        print(f"Successfully loaded {len(data)} rows from '{filename}'")
        return data
        
    except ValueError as e:
        print(f"Error parsing '{filename}': {e}")
        print("Check file format - should be numeric data")
        return None
        
    except Exception as e:
        print(f"Unexpected error loading '{filename}': {e}")
        return None

# Test
data = load_catalog_safe("nonexistent.txt")  # Handles missing file
data = load_catalog_safe("catalog.txt")  # Loads if exists
```

## Question 4
**Original question:** Organize a messy script into separate functions for load/process/plot/save

```python
# MESSY VERSION (everything in one block)
# import numpy as np
# import matplotlib.pyplot as plt
# data = np.loadtxt("flux.txt")
# clean = data[data > 100]
# mean = np.mean(clean)
# plt.hist(clean, bins=20)
# plt.savefig("plot.png")

# ORGANIZED VERSION
import numpy as np
import matplotlib.pyplot as plt

def load_data(filename):
    """Load flux data from file."""
    try:
        data = np.loadtxt(filename)
        print(f"Loaded {len(data)} measurements")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def process_data(data, threshold=100):
    """Filter data above threshold."""
    clean_data = data[data > threshold]
    print(f"Kept {len(clean_data)}/{len(data)} measurements")
    return clean_data

def analyze_data(data):
    """Calculate statistics."""
    stats = {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data, ddof=1),
        'min': np.min(data),
        'max': np.max(data)
    }
    return stats

def plot_data(data, filename='histogram.png'):
    """Create and save histogram."""
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=20, edgecolor='black')
    plt.xlabel('Flux (mJy)')
    plt.ylabel('Count')
    plt.title(f'Flux Distribution (N={len(data)})')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {filename}")

def save_results(stats, filename='results.txt'):
    """Save statistics to file."""
    with open(filename, 'w') as f:
        for key, value in stats.items():
            f.write(f"{key}: {value:.2f}\n")
    print(f"Saved results: {filename}")

def main():
    """Main analysis pipeline."""
    # Load
    data = load_data("flux.txt")
    if data is None:
        return
    
    # Process
    clean_data = process_data(data, threshold=100)
    
    # Analyze
    stats = analyze_data(clean_data)
    print("\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}")
    
    # Visualize
    plot_data(clean_data)
    
    # Save
    save_results(stats)

if __name__ == '__main__':
    main()
```

## Question 5
**Original question:** Add print statements to debug a calculation that's giving unexpected results

```python
def calculate_spectral_index_debug(flux1, freq1, flux2, freq2):
    """Calculate spectral index with debug output."""
    import math
    
    print("=== Debug: Spectral Index Calculation ===")
    print(f"Input flux1: {flux1}, freq1: {freq1}")
    print(f"Input flux2: {flux2}, freq2: {freq2}")
    
    # Check for invalid inputs
    if flux1 <= 0 or flux2 <= 0:
        print("ERROR: Flux values must be positive")
        return None
    if freq1 <= 0 or freq2 <= 0:
        print("ERROR: Frequency values must be positive")
        return None
    
    flux_ratio = flux1 / flux2
    print(f"Flux ratio: {flux_ratio:.4f}")
    
    freq_ratio = freq1 / freq2
    print(f"Frequency ratio: {freq_ratio:.4f}")
    
    log_flux = math.log(flux_ratio)
    log_freq = math.log(freq_ratio)
    print(f"log(flux ratio): {log_flux:.4f}")
    print(f"log(freq ratio): {log_freq:.4f}")
    
    alpha = log_flux / log_freq
    print(f"Spectral index α: {alpha:.4f}")
    print("==========================================")
    
    return alpha

# Test
alpha = calculate_spectral_index_debug(245.7, 144, 18.3, 1400)
```

---

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
