# Answers for Try This Section in 13 - Statistics as Functions

## Question 1
**Original question:** Write `calculate_variance()` that returns the variance

```python
def calculate_variance(values, sample=True):
    """Calculate variance (squared standard deviation)."""
    if len(values) < 2:
        return None
    
    mean = sum(values) / len(values)
    squared_diffs = [(x - mean) ** 2 for x in values]
    
    divisor = len(values) - 1 if sample else len(values)
    return sum(squared_diffs) / divisor

# Test
data = [245.7, 189.3, 312.5, 198.7, 267.4]
var = calculate_variance(data)
print(f"Variance: {var:.2f}")  # Output: Variance: 1914.99
```

## Question 2
**Original question:** Create a function that takes a list and returns a dict with just min, max, and range

```python
def get_range_stats(values):
    """Get min, max, and range of data."""
    if len(values) == 0:
        return None
    
    min_val = min(values)
    max_val = max(values)
    
    return {
        'min': min_val,
        'max': max_val,
        'range': max_val - min_val
    }

# Test
flux = [245.7, 189.3, 312.5, 198.7, 267.4]
stats = get_range_stats(flux)
print(stats)
# Output: {'min': 189.3, 'max': 312.5, 'range': 123.2}
```

## Question 3
**Original question:** Write `normalize_data(values)` that scales all values to 0-1 range

```python
def normalize_data(values):
    """
    Normalize values to 0-1 range.
    
    Formula: (x - min) / (max - min)
    """
    if len(values) == 0:
        return []
    
    min_val = min(values)
    max_val = max(values)
    
    if max_val == min_val:
        return [0.5] * len(values)  # All same value
    
    normalized = []
    for x in values:
        norm_x = (x - min_val) / (max_val - min_val)
        normalized.append(norm_x)
    
    return normalized

# Test
flux = [100, 150, 200, 250, 300]
norm = normalize_data(flux)
print(f"Original: {flux}")
print(f"Normalized: {[f'{x:.2f}' for x in norm]}")
# Output: Normalized: ['0.00', '0.25', '0.50', '0.75', '1.00']
```

## Question 4
**Original question:** Make a function that counts how many values are above the mean

```python
def count_above_mean(values):
    """Count how many values exceed the mean."""
    if len(values) == 0:
        return 0
    
    mean = sum(values) / len(values)
    count = 0
    for value in values:
        if value > mean:
            count += 1
    
    return count

# Test
data = [245.7, 189.3, 312.5, 198.7, 267.4]
mean = sum(data) / len(data)
above = count_above_mean(data)
print(f"Mean: {mean:.2f}")
print(f"Values above mean: {above} out of {len(data)}")
# Output: Values above mean: 2 out of 5
```

## Question 5
**Original question:** Write `remove_outliers_iqr(data)` using the IQR method instead of z-score

```python
def remove_outliers_iqr(data):
    """Remove outliers using IQR method."""
    if len(data) < 4:
        return data
    
    sorted_data = sorted(data)
    n = len(sorted_data)
    
    # Calculate Q1 and Q3
    q1_idx = n // 4
    q3_idx = 3 * n // 4
    q1 = sorted_data[q1_idx]
    q3 = sorted_data[q3_idx]
    
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Filter
    cleaned = [x for x in data if lower_bound <= x <= upper_bound]
    return cleaned

# Test
data = [245.7, 189.3, 312.5, 198.7, 10000.0, 267.4]
clean = remove_outliers_iqr(data)
print(f"Original: {len(data)} values")
print(f"After IQR filtering: {len(clean)} values")
print(f"Removed: {[x for x in data if x not in clean]}")
# Output: Removed: [10000.0]
```
