# Statistics as Functions

In the previous lesson, we calculated statistics manually. Now let's package them into reusable functions. This makes your code cleaner and easier to maintain - write the function once, use it everywhere. Change it in one place and all uses get updated.

## Why Functions for Statistics?

Imagine you need to calculate mean flux 20 times in your script for different subsets of data. Copying those 3 lines each time is tedious and error-prone. If you later realize you need to handle empty lists, you'd have to fix it in 20 places. With a function, fix it once:

```python
def calculate_mean(values):
    """Calculate the arithmetic mean of a list of numbers."""
    if len(values) == 0:
        return None  # Can't calculate mean of empty list
    return sum(values) / len(values)
```

Now you can use it anywhere: `mean = calculate_mean(flux_list)`. The `if len(values) == 0` check is **defensive programming** - preventing crashes from bad input.

## Building a Statistics Library

Let's create a set of functions that work together. Start with median:

```python
def calculate_median(values):
    """Calculate the median of a list of numbers."""
    if len(values) == 0:
        return None
    
    sorted_values = sorted(values)
    n = len(sorted_values)
    
    if n % 2 == 1:
        return sorted_values[n // 2]
    else:
        mid1 = sorted_values[n // 2 - 1]
        mid2 = sorted_values[n // 2]
        return (mid1 + mid2) / 2
```

And standard deviation with a parameter to choose sample vs population:

```python
def calculate_std(values, sample=True):
    """
    Calculate standard deviation.
    
    Parameters:
        values: list of numbers
        sample: if True, use N-1 (sample std). If False, use N (population std)
    
    Returns:
        Standard deviation, or None if fewer than 2 values.
    """
    if len(values) < 2:
        return None  # Need at least 2 values for meaningful std
    
    mean = sum(values) / len(values)
    squared_diffs = [(x - mean) ** 2 for x in values]
    
    divisor = len(values) - 1 if sample else len(values)
    variance = sum(squared_diffs) / divisor
    
    return variance ** 0.5
```

The `sample=True` parameter gives you flexibility. Most of the time you'll use the default (N-1), but you can override it if needed.

## A Complete Statistics Function

Often you want multiple statistics at once. Return them in a dictionary for clarity:

```python
def analyze_data(values):
    """
    Calculate comprehensive statistics for a dataset.
    
    Returns a dictionary with count, mean, median, std, min, max, and range.
    Returns None if list is empty.
    """
    if len(values) == 0:
        return None
    
    n = len(values)
    sorted_values = sorted(values)
    
    # Mean
    mean = sum(values) / n
    
    # Median
    if n % 2 == 1:
        median = sorted_values[n // 2]
    else:
        median = (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
    
    # Standard deviation (sample)
    if n > 1:
        squared_diffs = [(x - mean) ** 2 for x in values]
        std = (sum(squared_diffs) / (n - 1)) ** 0.5
    else:
        std = 0.0
    
    # Range
    min_val = min(values)
    max_val = max(values)
    
    return {
        'count': n,
        'mean': mean,
        'median': median,
        'std': std,
        'min': min_val,
        'max': max_val,
        'range': max_val - min_val
    }
```

Using it is clean:

```python
flux_list = [245.7, 189.3, 312.5, 198.7, 267.4]
stats = analyze_data(flux_list)

if stats:  # Check it's not None
    print(f"Mean: {stats['mean']:.2f} mJy")
    print(f"Median: {stats['median']:.2f} mJy")
    print(f"Std: {stats['std']:.2f} mJy")
```

## Functions for Quality Control

Statistics are often used for filtering. Here's an outlier detector using the z-score method (how many standard deviations from the mean):

```python
def is_outlier(value, data, threshold=3):
    """
    Check if a value is an outlier using z-score.
    
    Parameters:
        value: the value to check
        data: the full dataset  
        threshold: number of standard deviations (default 3)
    
    Returns:
        True if value is more than threshold std devs from mean.
    """
    if len(data) < 2:
        return False  # Can't determine with < 2 points
    
    mean = sum(data) / len(data)
    std = calculate_std(data)  # Uses our function from above
    
    if std == 0:
        return False  # All values identical
    
    z_score = abs((value - mean) / std)
    return z_score > threshold
```

Usage:

```python
flux_list = [245.7, 189.3, 312.5, 198.7, 267.4, 10000.0]  # Last one is suspicious
if is_outlier(10000.0, flux_list):
    print("10000.0 is an outlier - possibly bad data")
```

Here's a function that removes all outliers from a dataset:

```python
def remove_outliers(data, sigma=3):
    """
    Remove values beyond sigma standard deviations from mean.
    
    Returns:
        List with outliers removed.
    """
    if len(data) < 2:
        return data
    
    mean = sum(data) / len(data)
    std = calculate_std(data)
    
    if std == 0:
        return data  # All values identical
    
    cleaned = []
    for value in data:
        z_score = abs((value - mean) / std)
        if z_score <= sigma:
            cleaned.append(value)
    
    return cleaned
```

This is a common preprocessing step before calculating final statistics.

## Functions for Comparison

Sometimes you want to compare two datasets:

```python
def compare_datasets(data1, data2, labels=None):
    """Compare statistics of two datasets."""
    if labels is None:
        labels = ["Dataset 1", "Dataset 2"]
    
    stats1 = analyze_data(data1)
    stats2 = analyze_data(data2)
    
    if not stats1 or not stats2:
        print("One or both datasets are empty")
        return
    
    print(f"{labels[0]}:")
    print(f"  Mean: {stats1['mean']:.2f}, Std: {stats1['std']:.2f}")
    
    print(f"{labels[1]}:")
    print(f"  Mean: {stats2['mean']:.2f}, Std: {stats2['std']:.2f}")
    
    diff = stats1['mean'] - stats2['mean']
    print(f"\nMean difference: {diff:.2f}")

# Usage
epoch1_flux = [245.7, 189.3, 312.5]
epoch2_flux = [250.3, 195.8, 305.2]
compare_datasets(epoch1_flux, epoch2_flux, ["Epoch 1", "Epoch 2"])
```

## Best Practices for Statistical Functions

**Functions should do one thing well.** Don't make a function that calculates statistics AND plots them AND saves to a file. Separate concerns:

```python
# Good - each function has one job
stats = analyze_data(flux)
plot_histogram(flux)
save_to_file(stats, "results.txt")
```

**Return values, don't print.** Printing inside functions makes them less reusable:

```python
# Bad - can't use the value later
def calculate_mean(values):
    mean = sum(values) / len(values)
    print(f"Mean: {mean}")  # Prints immediately!

# Good - returns the value for later use
def calculate_mean(values):
    return sum(values) / len(values)
```

**Handle edge cases.** Empty lists, single values, all-identical values - these break naive implementations. Check for them:

```python
def calculate_std(values):
    if len(values) < 2:
        return None  # Or raise ValueError("Need at least 2 values")
    # ... rest of function
```

**Document what can be None.** Let users know when functions might return None:

```python
def calculate_std(values):
    """
    Calculate standard deviation.
    
    Returns:
        Standard deviation as float, or None if fewer than 2 values.
    """
```

## Things Worth Noting

**Don't use mutable default arguments.** This is a Python gotcha:

```python
# Bad - list gets shared between calls!
def add_to_list(item, my_list=[]):
    my_list.append(item)
    return my_list

# Good
def add_to_list(item, my_list=None):
    if my_list is None:
        my_list = []
    my_list.append(item)
    return my_list
```

**Type hints can help.** For larger projects, consider adding type hints (we haven't covered these, but they're useful):

```python
def calculate_mean(values: list) -> float:
    return sum(values) / len(values)
```

## Try This

1. Write `calculate_variance()` that returns variance (before taking square root)
2. Create `get_range(values)` that returns a dict with min, max, and range
3. Write `normalize_data(values)` that scales all values to 0-1 range
4. Make `count_above_mean(values)` that returns how many values exceed the mean
5. Write `remove_outliers_iqr(data)` using the IQR method instead of z-score

## How This Is Typically Used in Astronomy

Professional astronomers package their analysis code into reusable functions, creating personal libraries they import across projects. This ensures consistency, makes code testable, and allows sharing with collaborators.

## Related Lessons

**Previous**: [12_statistics_from_scratch.md](12_statistics_from_scratch.md) - Manual calculations

**Next**: [14_numpy_basics.md](14_numpy_basics.md) - NumPy does all this much faster

**Uses**: [10_functions.md](10_functions.md) - How to write functions
