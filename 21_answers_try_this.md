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
