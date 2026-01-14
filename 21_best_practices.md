# Best Practices

You can write Python that works. But can you read it six months later? Can colleagues understand it? Best practices make code maintainable, debuggable, and shareable.

## Writing Clear Code

**Use descriptive variable names.** Don't make readers guess:

```python
# Bad
x = 245.7
y = 144
z = x * (y/1400)**-0.7

# Good
flux_144MHz = 245.7
reference_freq = 144
target_freq = 1400
spectral_index = -0.7
flux_target = flux_144MHz * (reference_freq/target_freq)**spectral_index
```

Extra characters don't slow your code. Clarity matters more than brevity.

**Break complex expressions into steps:**

```python
# Hard to read
result = ((flux1/flux2) / (np.sqrt((err1/flux1)**2 + (err2/flux2)**2))) > 3

# Clearer
flux_ratio = flux1 / flux2
relative_error = np.sqrt((err1/flux1)**2 + (err2/flux2)**2)
significance = flux_ratio / relative_error
is_significant = significance > 3
```

Each line has one clear purpose. Debugging is easier because you can print intermediate values.

## Comments and Documentation

**Comment the why, not the what:**

```python
# Bad - comment just repeats code
flux = flux * 1000  # Multiply flux by 1000

# Good - explains reasoning
flux = flux * 1000  # Convert Jy to mJy for catalog format
```

The code already shows what happens. Comments should explain why you made that choice.

**Use docstrings for functions:**

```python
def calculate_spectral_index(flux1, freq1, flux2, freq2):
    """
    Calculate spectral index from two flux measurements.
    
    Uses the power law S ∝ ν^α to determine α.
    
    Parameters:
        flux1, flux2: Flux densities in same units
        freq1, freq2: Frequencies in same units
        
    Returns:
        Spectral index α
        
    Example:
        >>> alpha = calculate_spectral_index(245.7, 144, 18.3, 1400)
        >>> print(alpha)
        -1.112
    """
    import math
    return math.log(flux1/flux2) / math.log(freq1/freq2)
```

Good documentation helps future you and anyone else using your code.

## Organizing Code into Scripts

Once code works in a notebook, turn it into a reusable script:

```python
# analyze_sources.py

import numpy as np
import matplotlib.pyplot as plt

def load_catalog(filename):
    """Load source catalog from CSV."""
    data = np.genfromtxt(filename, delimiter=',', names=True)
    return data

def apply_quality_cuts(data):
    """Filter to good quality sources."""
    mask = data['quality_flag'] == 0
    return data[mask]

def plot_flux_distribution(flux, filename='flux_hist.png'):
    """Create and save histogram of flux distribution."""
    plt.figure(figsize=(8, 6))
    plt.hist(flux, bins=30, edgecolor='black')
    plt.xlabel('Flux (mJy)')
    plt.ylabel('Count')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main analysis pipeline."""
    # Load data
    data = load_catalog('catalog.csv')
    print(f"Loaded {len(data)} sources")
    
    # Apply cuts
    clean_data = apply_quality_cuts(data)
    print(f"{len(clean_data)} sources after quality cuts")
    
    # Analyze
    flux = clean_data['flux']
    mean_flux = np.mean(flux)
    median_flux = np.median(flux)
    
    print(f"Mean flux: {mean_flux:.2f} mJy")
    print(f"Median flux: {median_flux:.2f} mJy")
    
    # Plot
    plot_flux_distribution(flux)
    print("Saved flux_hist.png")

if __name__ == '__main__':
    main()
```

The `if __name__ == '__main__'` block lets you import functions from this script without running the analysis automatically.

Run it from terminal:

```bash
python3 analyze_sources.py
```

## Error Handling

Anticipate what can go wrong and handle it gracefully:

```python
def load_catalog(filename):
    """Load catalog, handling common errors."""
    import numpy as np
    import os
    
    # Check if file exists
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Catalog file not found: {filename}")
    
    # Try to load
    try:
        data = np.genfromtxt(filename, delimiter=',', names=True)
    except ValueError as e:
        raise ValueError(f"Error parsing {filename}: {e}")
    
    # Validate
    if len(data) == 0:
        raise ValueError(f"Catalog {filename} is empty")
    
    return data
```

This gives helpful error messages instead of cryptic tracebacks.

For non-critical errors, use warnings:

```python
import warnings

def calculate_spectral_index(flux1, flux2, freq1, freq2):
    if flux1 <= 0 or flux2 <= 0:
        warnings.warn("Negative or zero flux - spectral index undefined")
        return None
    # ... rest of calculation
```

## Debugging Techniques

**Print statements are your friend:**

```python
def complex_calculation(data):
    step1 = data * 2
    print(f"After step 1: {step1[:5]}")  # Show first 5 values
    
    step2 = np.log(step1)
    print(f"After step 2: {step2[:5]}")
    
    result = np.sum(step2)
    print(f"Final result: {result}")
    return result
```

Add prints at each step to see where things go wrong.

**Use assertions to catch assumptions:**

```python
def process_flux(flux):
    assert len(flux) > 0, "Flux array is empty"
    assert np.all(flux > 0), "Flux contains non-positive values"
    # ... continue processing
```

Assertions fail loudly when assumptions break.

**Check intermediate results:**

```python
# Calculate something
result = complex_function(data)

# Sanity check
assert not np.any(np.isnan(result)), "Result contains NaN"
assert not np.any(np.isinf(result)), "Result contains infinity"
assert np.all(result >= 0), "Result has negative values"
```

## Version Control (Brief Introduction)

Once you have working code, protect it with version control. Git tracks changes:

```bash
# Initialize repository
git init

# Add files
git add analyze_sources.py

# Commit with message
git commit -m "Initial analysis script"

# Later, after changes
git add analyze_sources.py
git commit -m "Added quality cuts"
```

This creates a history. You can always go back to previous versions if something breaks.

For collaboration, use GitHub or GitLab to share code.

## Code Organization Patterns

**Separate concerns:**

```python
# Good structure
load_data()       # I/O
clean_data()      # Preprocessing
analyze_data()    # Calculations
plot_results()    # Visualization
save_results()    # Output
```

Each function has one job. Easy to test, debug, and modify.

**Don't repeat yourself (DRY):**

```python
# Bad - repeated code
mean_flux_A = np.mean(flux_A)
std_flux_A = np.std(flux_A)
mean_flux_B = np.mean(flux_B)
std_flux_B = np.std(flux_B)
mean_flux_C = np.mean(flux_C)
std_flux_C = np.std(flux_C)

# Good - use a function
def get_stats(flux):
    return np.mean(flux), np.std(flux)

mean_A, std_A = get_stats(flux_A)
mean_B, std_B = get_stats(flux_B)
mean_C, std_C = get_stats(flux_C)
```

## Things Worth Noting

**Premature optimization is the enemy.** Write clear code first, optimize later if needed. NumPy is already fast - most astronomy code isn't CPU-bound anyway.

**Configuration files for parameters:** Don't hardcode paths and thresholds. Use a config file:

```python
# config.py
CATALOG_PATH = '/data/lotss/dr2_catalog.csv'
FLUX_THRESHOLD = 100.0  # mJy
QUALITY_FLAG_MAX = 1
```

Then import: `from config import CATALOG_PATH`

**Testing matters:** Write simple tests for your functions:

```python
def test_spectral_index():
    # Known values
    alpha = calculate_spectral_index(100, 100, 100, 1000)
    assert abs(alpha - 0.0) < 0.001  # Should be 0 for flat spectrum
    
    alpha = calculate_spectral_index(100, 100, 10, 1000)
    assert alpha < 0  # Should be negative for decreasing flux
```

Run tests before committing changes.

**Notebook vs script:** Notebooks are great for exploration. Once analysis is solid, convert to a script for reproducibility.

## Try This

1. Take one of your earlier scripts and add descriptive variable names everywhere
2. Write docstrings for 3 functions you've created
3. Add error handling to a file-loading function (check if file exists, handle parse errors)
4. Organize a messy script into separate functions for load/process/plot/save
5. Add print statements to debug a calculation that's giving unexpected results

## How This Is Typically Used in Astronomy

Professional astronomy code follows these practices. Clean, documented code gets cited in papers, shared with collaborators, and reused for future projects. Messy code gets abandoned.

Investment in readability pays off when you revisit analysis months later or when reviewers ask to see your code.

## Related Lessons

**Previous**: [20_working_with_real_data.md](20_working_with_real_data.md) - Practical data handling

**Next**: [22_advanced_topics.md](22_advanced_topics.md) - Going further

**Foundation**: All previous lessons - best practices apply to everything
