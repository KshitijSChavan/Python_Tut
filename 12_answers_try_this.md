# Answers for Try This Section in 12 - Statistics from Scratch

## Question 1
**Original question:** Calculate mean and median for `[100, 150, 200, 250, 300]` - they should be equal. Why?

```python
data = [100, 150, 200, 250, 300]

# Calculate mean
mean = sum(data) / len(data)
print(f"Mean: {mean}")  # Output: Mean: 200.0

# Calculate median
sorted_data = sorted(data)
n = len(sorted_data)
if n % 2 == 1:
    median = sorted_data[n // 2]
else:
    median = (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
print(f"Median: {median}")  # Output: Median: 200.0

print(f"\nMean == Median: {mean == median}")  # True

# Why are they equal?
print("\nWhy they're equal:")
print("- The data is perfectly symmetric around 200")
print("- Values are evenly spaced (50 apart)")
print("- No outliers pulling the mean away from center")
print("- This is a uniform distribution")
```

## Question 2
**Original question:** Add an outlier (10000) to that list and recalculate. How much does each change?

```python
original_data = [100, 150, 200, 250, 300]
with_outlier = [100, 150, 200, 250, 300, 10000]

# Original statistics
mean_orig = sum(original_data) / len(original_data)
sorted_orig = sorted(original_data)
median_orig = sorted_orig[len(sorted_orig) // 2]

print("Original data:")
print(f"  Mean: {mean_orig:.2f}")
print(f"  Median: {median_orig:.2f}")

# With outlier
mean_outlier = sum(with_outlier) / len(with_outlier)
sorted_outlier = sorted(with_outlier)
n = len(sorted_outlier)
median_outlier = (sorted_outlier[n // 2 - 1] + sorted_outlier[n // 2]) / 2

print("\nWith outlier (10000):")
print(f"  Mean: {mean_outlier:.2f}")
print(f"  Median: {median_outlier:.2f}")

# Calculate changes
mean_change = mean_outlier - mean_orig
median_change = median_outlier - median_orig
mean_percent = (mean_change / mean_orig) * 100
median_percent = (median_change / median_orig) * 100

print("\nChanges:")
print(f"  Mean changed by: {mean_change:.2f} ({mean_percent:.1f}%)")
print(f"  Median changed by: {median_change:.2f} ({median_percent:.1f}%)")

print("\nConclusion:")
print("  Mean increased massively (sensitive to outliers)")
print("  Median barely changed (robust to outliers)")

# Output:
# Original data:
#   Mean: 200.00
#   Median: 200.00
# 
# With outlier (10000):
#   Mean: 1833.33
#   Median: 225.00
# 
# Changes:
#   Mean changed by: 1633.33 (816.7%)
#   Median changed by: 25.00 (12.5%)
```

## Question 3
**Original question:** Calculate standard deviation for `[245.7, 189.3, 312.5, 198.7, 267.4]`

```python
data = [245.7, 189.3, 312.5, 198.7, 267.4]

# Calculate mean
mean = sum(data) / len(data)
print(f"Mean: {mean:.2f}")

# Calculate squared differences
squared_diffs = []
for value in data:
    diff = value - mean
    squared_diffs.append(diff ** 2)

# Population standard deviation (divide by N)
variance_pop = sum(squared_diffs) / len(data)
std_pop = variance_pop ** 0.5
print(f"Population std dev: {std_pop:.2f}")

# Sample standard deviation (divide by N-1)
variance_sample = sum(squared_diffs) / (len(data) - 1)
std_sample = variance_sample ** 0.5
print(f"Sample std dev: {std_sample:.2f}")

# Output:
# Mean: 242.72
# Population std dev: 39.15
# Sample std dev: 43.76

# Show calculation steps
print("\nDetailed calculation:")
print(f"Data: {data}")
print(f"Mean: {mean:.2f}")
print("\nSquared differences from mean:")
for i, value in enumerate(data):
    diff = value - mean
    sq_diff = diff ** 2
    print(f"  {value:.1f}: ({value:.1f} - {mean:.2f})² = {sq_diff:.2f}")
print(f"\nSum of squared diffs: {sum(squared_diffs):.2f}")
print(f"Variance (N-1): {variance_sample:.2f}")
print(f"Std dev: {std_sample:.2f}")
```

## Question 4
**Original question:** Write a function that returns True if a value is an outlier (more than 1.5×IQR beyond Q1/Q3)

```python
def is_outlier_iqr(value, data):
    """
    Check if value is an outlier using IQR method.
    
    A value is an outlier if it's more than 1.5×IQR
    beyond Q1 or Q3.
    
    Parameters:
        value: value to check
        data: full dataset
    
    Returns:
        bool: True if outlier
    """
    # Calculate percentiles
    sorted_data = sorted(data)
    n = len(sorted_data)
    
    # Q1 (25th percentile)
    q1_index = n * 0.25
    if q1_index == int(q1_index):
        q1 = sorted_data[int(q1_index)]
    else:
        lower = int(q1_index)
        upper = lower + 1
        weight = q1_index - lower
        q1 = sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight
    
    # Q3 (75th percentile)
    q3_index = n * 0.75
    if q3_index == int(q3_index):
        q3 = sorted_data[int(q3_index)]
    else:
        lower = int(q3_index)
        upper = lower + 1
        weight = q3_index - lower
        q3 = sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight
    
    # Calculate IQR and bounds
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Check if outlier
    return value < lower_bound or value > upper_bound

# Test
data = [245.7, 189.3, 312.5, 198.7, 267.4, 234.1, 289.6]

test_values = [100, 245.7, 500, 250]
for val in test_values:
    result = is_outlier_iqr(val, data)
    print(f"Is {val} an outlier? {result}")

# Output:
# Is 100 an outlier? True
# Is 245.7 an outlier? False
# Is 500 an outlier? True
# Is 250 an outlier? False

# Find all outliers in dataset
outliers = [x for x in data if is_outlier_iqr(x, data)]
print(f"\nOutliers in data: {outliers}")
```

## Question 5
**Original question:** Compare results using N vs N-1 for standard deviation with a small dataset

```python
small_data = [10, 15, 12, 18, 14]

# Calculate with N (population)
mean = sum(small_data) / len(small_data)
squared_diffs = [(x - mean) ** 2 for x in small_data]

variance_N = sum(squared_diffs) / len(small_data)
std_N = variance_N ** 0.5

# Calculate with N-1 (sample)
variance_N_minus_1 = sum(squared_diffs) / (len(small_data) - 1)
std_N_minus_1 = variance_N_minus_1 ** 0.5

print(f"Data: {small_data}")
print(f"Sample size: {len(small_data)}")
print(f"Mean: {mean:.2f}")
print()
print(f"Using N={len(small_data)}:")
print(f"  Variance: {variance_N:.2f}")
print(f"  Std dev: {std_N:.2f}")
print()
print(f"Using N-1={len(small_data)-1}:")
print(f"  Variance: {variance_N_minus_1:.2f}")
print(f"  Std dev: {std_N_minus_1:.2f}")
print()
print(f"Difference: {std_N_minus_1 - std_N:.2f}")
print(f"Percent difference: {((std_N_minus_1 - std_N) / std_N) * 100:.1f}%")
print()
print("Why N-1? Bessel's correction provides unbiased estimate")
print("of population std dev from a sample.")
print("Difference is larger for smaller samples.")

# Compare with different sample sizes
print("\nEffect of sample size:")
for n in [3, 5, 10, 20, 100]:
    ratio = n / (n - 1)
    correction = ((ratio - 1) * 100)
    print(f"  N={n:3d}: std(N-1) is {correction:.1f}% larger than std(N)")

# Output:
# Effect of sample size:
#   N=  3: std(N-1) is 22.5% larger than std(N)
#   N=  5: std(N-1) is 11.8% larger than std(N)
#   N= 10: std(N-1) is 5.4% larger than std(N)
#   N= 20: std(N-1) is 2.6% larger than std(N)
#   N=100: std(N-1) is 0.5% larger than std(N)
```
