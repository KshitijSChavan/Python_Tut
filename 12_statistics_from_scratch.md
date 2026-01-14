# Statistics from Scratch

Statistical measures like mean, median, and standard deviation are fundamental to analyzing astronomical data. Let's calculate them manually to understand what's actually happening. Later we'll use NumPy to do it faster, but understanding the mechanics helps you know when something goes wrong.

## Mean - The Average

The mean is straightforward: add everything up, divide by how many values you have. It's the most common measure of central tendency:

```python
flux_measurements = [245.7, 189.3, 312.5, 198.7, 267.4]

total = sum(flux_measurements)
mean = total / len(flux_measurements)
print(f"Mean: {mean:.2f} mJy")  # Output: Mean: 242.72 mJy
```

The mean has a weakness though - it's sensitive to outliers. If you have measurements `[10, 20, 30, 40, 10000]`, the mean is 2020, which doesn't represent the typical value at all. This is where median becomes useful.

## Median - The Middle Value

The median is the middle value when you sort your data. Half your measurements are above it, half below:

```python
flux_measurements = [245.7, 189.3, 312.5, 198.7, 267.4]

sorted_flux = sorted(flux_measurements)
n = len(sorted_flux)

if n % 2 == 1:
    # Odd number - take the middle value
    median = sorted_flux[n // 2]
else:
    # Even number - average the two middle values
    mid1 = sorted_flux[n // 2 - 1]
    mid2 = sorted_flux[n // 2]
    median = (mid1 + mid2) / 2

print(f"Median: {median:.2f} mJy")
```

Why the odd/even split? With 5 values, the middle is index 2 (the 3rd value). With 4 values, there's no single middle, so we average indices 1 and 2. The formula `n // 2` handles this automatically for odd numbers.

The median is **robust to outliers**. That dataset `[10, 20, 30, 40, 10000]` has a median of 30 - much more representative than the mean of 2020.

## Variance - Measuring Spread

Variance quantifies how spread out your data is. The calculation: for each value, find how far it is from the mean, square that distance, then average those squared distances:

```python
flux_measurements = [245.7, 189.3, 312.5, 198.7, 267.4]

mean = sum(flux_measurements) / len(flux_measurements)

squared_differences = []
for flux in flux_measurements:
    diff = flux - mean
    squared_differences.append(diff ** 2)

variance = sum(squared_differences) / len(squared_differences)
print(f"Variance: {variance:.2f}")  # Output: Variance: 1914.99
```

Why square the differences? If we didn't, positive and negative differences would cancel out, giving zero variance even for spread-out data. Squaring makes everything positive.

The units are squared though (mJy²), which is hard to interpret. That's why we typically use standard deviation instead.

## Standard Deviation - Variance in Original Units

Standard deviation is just the square root of variance, bringing the units back to the original scale:

```python
std_dev = variance ** 0.5
print(f"Standard deviation: {std_dev:.2f} mJy")  # Output: Standard deviation: 43.76 mJy
```

Think of standard deviation as "the typical distance from the mean". For normally distributed data, about 68% of values fall within one standard deviation of the mean.

Here's the complete calculation in one block:

```python
def calculate_std(values):
    mean = sum(values) / len(values)
    squared_diffs = [(x - mean) ** 2 for x in values]
    variance = sum(squared_diffs) / len(values)
    return variance ** 0.5

flux = [245.7, 189.3, 312.5, 198.7, 267.4]
std = calculate_std(flux)
print(f"Std dev: {std:.2f} mJy")
```

## Sample vs Population Standard Deviation

There's a subtle but important distinction. When calculating standard deviation for a **sample** (not the entire population), you should divide by N-1 instead of N:

```python
def sample_std(values):
    mean = sum(values) / len(values)
    squared_diffs = [(x - mean) ** 2 for x in values]
    variance = sum(squared_diffs) / (len(values) - 1)  # N-1 here!
    return variance ** 0.5
```

This is called **Bessel's correction**. It gives an unbiased estimate of the population standard deviation from a sample. In astronomy, you're usually analyzing samples (observed sources) not entire populations (all sources in the universe), so use N-1.

The difference is small for large datasets but matters for small samples. With 5 measurements, dividing by 4 instead of 5 increases your estimate by about 12%.

## Percentiles

Percentiles tell you what value a certain percentage of data falls below. The 25th percentile (Q1) means 25% of values are below it. The 50th percentile is the median. The 75th percentile (Q3) means 75% are below it:

```python
def calculate_percentile(data, percentile):
    sorted_data = sorted(data)
    n = len(sorted_data)
    position = (percentile / 100) * (n - 1)
    
    if position == int(position):
        return sorted_data[int(position)]
    else:
        # Interpolate between two values
        lower = int(position)
        upper = lower + 1
        weight = position - lower
        return sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight

flux = [245.7, 189.3, 312.5, 198.7, 267.4]
q1 = calculate_percentile(flux, 25)
q3 = calculate_percentile(flux, 75)
print(f"Q1: {q1:.2f}, Q3: {q3:.2f}")
```

The **IQR (interquartile range)** is Q3 - Q1. It's useful for detecting outliers: values more than 1.5×IQR beyond Q1 or Q3 are often considered outliers.

## Things Worth Noting

**Empty lists cause division by zero:**

```python
# values = []
# mean = sum(values) / len(values)  # ZeroDivisionError!
```

Always check if your list has data before calculating statistics.

**Mean vs median for skewed data:** If your data has a long tail (many small values, few huge ones), the mean gets pulled toward the tail. The median better represents the "typical" value. In astronomy, flux distributions are often skewed, so median is frequently more useful than mean.

**Floating point precision:** Computers can't represent all decimals exactly. You might get `0.30000000000000004` instead of `0.3`. This is normal and usually doesn't matter for astronomical measurements.

**Standard deviation and normal distributions:** The "68% within 1 std dev" rule only applies to bell-curve (normal) distributions. Many astronomical quantities aren't normally distributed.

## Try This

1. Calculate mean and median for `[100, 150, 200, 250, 300]` - they should be equal. Why?
2. Add an outlier (10000) to that list and recalculate. How much does each change?
3. Calculate standard deviation using both N and N-1 for a small dataset and compare
4. Find the IQR for your flux measurements
5. Write a function that returns True if a value is more than 1.5×IQR beyond Q1/Q3 (outlier detection)

## How This Is Typically Used in Astronomy

Calculating average flux across observations, determining measurement uncertainties (std dev), identifying outliers in catalogs, characterizing source populations (median properties), and making quality cuts (reject if more than 3σ from mean).

Next we'll package these calculations into reusable functions, then learn how NumPy does all this instantly.

## Related Lessons

**Previous**: [11_file_io_and_csv.md](11_file_io_and_csv.md) - Reading data files

**Next**: [13_statistics_as_functions.md](13_statistics_as_functions.md) - Making these reusable

**Later**: [14_numpy_basics.md](14_numpy_basics.md) - NumPy has built-in `np.mean()`, `np.std()`, etc.
