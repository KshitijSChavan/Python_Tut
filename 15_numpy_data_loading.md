# NumPy Data Loading and Analysis

Reading data files is one of the first steps in any analysis. NumPy makes loading numerical data much easier and faster than parsing CSV manually.

## Loading Simple Files

For files that are just columns of numbers, `np.loadtxt()` is straightforward. Say you have a file `sources.txt` with one flux value per line - just point NumPy at it:

```python
import numpy as np
fluxes = np.loadtxt("sources.txt")
print(fluxes)  # Array of all the flux values
```

If your file has multiple columns (like RA, Dec, Flux), NumPy loads it as a 2D array:

```python
data = np.loadtxt("catalog.txt")  # Each row is a source
ra = data[:, 0]  # All values from first column
dec = data[:, 1]  # All values from second column  
flux = data[:, 2]  # All values from third column
```

That `:, 0` notation means "all rows, column 0" - much cleaner than looping through rows.

## Handling Headers

Many files have a header row with column names. Skip it with `skiprows=1`:

```python
# File has: RA Dec Flux (header row)
# then data rows
data = np.loadtxt("catalog.txt", skiprows=1)
```

If the file is more complicated, use `np.genfromtxt()` instead - it's more forgiving about formatting issues.

## CSV Files and Column Selection

For CSV files (comma-separated), specify the delimiter. If some columns are text (like source names), use `usecols` to load only the numeric ones:

```python
# File: name,ra,dec,flux
# Skip column 0 (name) since it's text, load columns 1-3
data = np.loadtxt("sources.csv", delimiter=",", skiprows=1, usecols=(1,2,3))
```

Column 0 is the first column, column 1 is the second, etc. This saves memory and avoids errors from trying to load text as numbers.

## Dealing with Missing Data

Real datasets have gaps - missing measurements, bad observations, corrupted values. These show up as `NaN` (Not a Number). NumPy can handle them:

```python
data = np.genfromtxt("data_with_gaps.txt")  # Loads NaN for missing values
clean_data = data[~np.isnan(data)]  # Keep only non-NaN values
```

The `~` means "not", so `~np.isnan(data)` translates to "not NaN" - giving you only valid measurements. This is essential for quality control.

## Filtering and Quality Cuts

Once data is loaded, you'll want to apply quality cuts. NumPy's boolean indexing makes this elegant:

```python
flux = np.loadtxt("fluxes.txt")

# Keep only detected sources (> 100 mJy)
detected = flux[flux > 100]
print(f"Detected {len(detected)} out of {len(flux)} sources")

# Multiple conditions - bright but not saturated
valid = flux[(flux > 200) & (flux < 10000)]

# Count how many meet a criterion (True counts as 1)
num_bright = np.sum(flux > 500)
```

This replaces loops with fast array operations. For a million sources, this is orders of magnitude faster than looping.

## Statistical Analysis of Loaded Data

With data loaded, calculating statistics is one line per statistic:

```python
flux = np.loadtxt("catalog.txt", usecols=2)  # Load flux column

mean_flux = np.mean(flux)
median_flux = np.median(flux)
std_flux = np.std(flux, ddof=1)

# Percentiles for robust statistics
q25, q75 = np.percentile(flux, [25, 75])
iqr = q75 - q25

print(f"Median: {median_flux:.2f} mJy, IQR: {iqr:.2f} mJy")
```

The IQR (interquartile range) is often more meaningful than standard deviation when you have outliers, which is common in astronomy.

## Identifying Outliers

The sigma-clipping method: anything more than 3 standard deviations from the mean is suspect:

```python
mean = np.mean(flux)
std = np.std(flux, ddof=1)
outliers = flux[np.abs(flux - mean) > 3 * std]
print(f"Found {len(outliers)} potential outliers")
```

This identifies extreme values that might be instrumental artifacts or genuinely interesting transient sources.

## Creating Histograms

A histogram shows the distribution of your data by counting how many values fall in each bin. NumPy's `np.histogram()` does the counting:

```python
flux = np.array([245.7, 189.3, 312.5, 198.7, 267.4, 156.2, 289.3])
counts, bin_edges = np.histogram(flux, bins=3)
```

This returns two things: `counts` tells you how many sources in each bin, `bin_edges` tells you the boundaries. Later we'll use matplotlib to actually plot histograms, but this gives you the raw data.

## Simple Linear Fitting

For power-law relationships (common in astronomy), fit in log space. If you have flux vs frequency data and want the spectral index:

```python
freq = np.array([144, 323, 608, 1400])  # MHz
flux = np.array([245.7, 189.3, 156.2, 98.5])  # mJy

# Fit log(flux) vs log(freq) - gives spectral index
coeffs = np.polyfit(np.log(freq), np.log(flux), 1)
spectral_index = coeffs[0]
print(f"Spectral index α = {spectral_index:.3f}")
```

The slope from this log-log fit is your spectral index α, since flux follows S ∝ ν^α.

## Saving Results

After analyzing data, save your results:

```python
# Simple save
np.savetxt("output.txt", flux)

# With formatting (2 decimal places)
np.savetxt("output.txt", flux, fmt="%.2f")

# Multiple columns with header
data = np.column_stack([ra, dec, flux])
np.savetxt("catalog.txt", data, header="RA Dec Flux", fmt="%.4f")
```

The `fmt` parameter controls how numbers are written - `%.2f` means 2 decimal places, `%.4f` means 4 decimal places.

## Things Worth Noting

**loadtxt vs genfromtxt:** Use `loadtxt` for clean data (it's faster). Use `genfromtxt` if you have missing values or formatting issues (it's more forgiving but slower).

**Memory with huge files:** Loading a file with millions of rows all at once uses a lot of RAM. If you hit memory limits, you'll need to process the file in chunks - we won't cover that here but it's good to know the limitation exists.

**File not found errors:** If you get `FileNotFoundError`, check that the file is in the directory where Python is running. Use absolute paths if needed: `/home/user/data/catalog.txt` instead of just `catalog.txt`.

**Different delimiters:** Files don't always use spaces. For tabs, use `delimiter="\t"`. For any whitespace, use `delimiter=None`. For commas (CSV), use `delimiter=","`.

**Checking for NaN:** Use `np.any(np.isnan(data))` to check if there are any NaN values, or `np.all(np.isnan(data))` to check if everything is NaN. Helpful for data validation.

## Try This

1. Create a text file with 5 flux values and load it with NumPy
2. Load a multi-column file and calculate the mean of one column
3. Load data and filter to keep only values > 200
4. Load a file with missing values (write some as NaN) and remove them
5. Calculate percentiles of loaded data and identify values in the top 10%

## How This Is Typically Used in Astronomy

Loading LoTSS, FIRST, or NVSS catalogs, reading photometry measurements, processing spectroscopic data, applying quality cuts to thousands of sources, calculating population statistics, and preparing data for visualization.

The combination of fast loading and boolean indexing makes NumPy essential for survey work.

## Related Lessons

**Previous**: [14_numpy_basics.md](14_numpy_basics.md) - NumPy arrays and operations

**Next**: [16_matplotlib_basics.md](16_matplotlib_basics.md) - Visualizing your data

**Alternative**: [11_file_io_and_csv.md](11_file_io_and_csv.md) - Python's csv module for mixed data types
