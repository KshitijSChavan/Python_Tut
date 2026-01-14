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
print(f"  Sigma method removed: {sorted(data[~np.isin(data, sigma_cleaned)])}")
print(f"  IQR method removed: {sorted(data[~np.isin(data, iqr_cleaned)])}")
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
import matplotlib.pyplot as plt

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
