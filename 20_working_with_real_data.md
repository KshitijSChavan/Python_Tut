# Working with Real Data

Real astronomical data is messy. Files have headers and metadata, values are missing, there are outliers and bad measurements, columns need matching between catalogs. Let's learn practical techniques for handling these challenges.

## Loading Data with Headers and Comments

Real catalog files often have metadata at the top:

```
# LoTSS DR2 Source Catalog
# Generated: 2023-04-15
# Columns: name, ra, dec, flux_144, flux_error
J1225+4011,187.7056,12.3911,245.7,10.2
J1445+3131,221.2134,31.5203,312.5,12.8
J0958+3224,149.5421,32.4156,198.7,9.5
```

NumPy can skip comment lines:

```python
import numpy as np

data = np.loadtxt('catalog.csv', delimiter=',', comments='#')
names = data[:, 0]  # Wait - this fails! Column 0 is text
```

Problem: the name column is text. Use `genfromtxt` with dtype for mixed types:

```python
data = np.genfromtxt('catalog.csv', delimiter=',', comments='#',
                     dtype=None, names=True, encoding='utf-8')

print(data['ra'])    # Access by column name
print(data['flux_144'])
```

The `dtype=None` auto-detects types, `names=True` uses the last comment line as column names.

## Handling Missing Data

Real datasets have gaps - measurements failed, sources weren't detected at all frequencies:

```
name,ra,dec,flux_144,flux_1400
J1225+4011,187.7,12.3,245.7,18.3
J1445+3131,221.2,31.5,312.5,
J0958+3224,149.5,32.4,,15.2
```

Load with `genfromtxt` - it converts missing values to NaN:

```python
import numpy as np

data = np.genfromtxt('catalog.csv', delimiter=',', skip_header=1, 
                     filling_values=np.nan, names=True)

flux_144 = data['flux_144']
flux_1400 = data['flux_1400']

# Which sources have both measurements?
has_both = ~np.isnan(flux_144) & ~np.isnan(flux_1400)
valid_144 = flux_144[has_both]
valid_1400 = flux_1400[has_both]

print(f"{np.sum(has_both)} sources with measurements at both frequencies")
```

The tilde `~` means "not", so `~np.isnan()` gives True for valid numbers.

## Cleaning Data - Removing Outliers

Before analysis, identify and handle outliers:

```python
import numpy as np

def remove_outliers_sigma(data, sigma=3):
    """Remove points beyond sigma standard deviations from median."""
    # Use median instead of mean (more robust)
    median = np.median(data)
    std = np.std(data)
    
    # Create mask for valid data
    distance = np.abs(data - median)
    mask = distance < sigma * std
    
    return data[mask], mask

flux = np.array([245.7, 189.3, 312.5, 198.7, 9999.0, 267.4])  # One bad value
clean_flux, mask = remove_outliers_sigma(flux)

print(f"Original: {len(flux)} values")
print(f"After cleaning: {len(clean_flux)} values")
print(f"Removed: {flux[~mask]}")
```

Alternatively, use the IQR (interquartile range) method:

```python
def remove_outliers_iqr(data):
    """Remove points beyond 1.5 * IQR from quartiles."""
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    
    mask = (data >= lower) & (data <= upper)
    return data[mask], mask
```

The IQR method is more robust when you have extreme outliers that skew the standard deviation.

## Cross-Matching Catalogs

You have two catalogs and want to find common sources. Match by position (RA, Dec):

```python
import numpy as np

def simple_crossmatch(ra1, dec1, ra2, dec2, tolerance=5.0):
    """
    Find matches between two catalogs.
    
    Parameters:
        ra1, dec1: coordinates of catalog 1 (degrees)
        ra2, dec2: coordinates of catalog 2 (degrees)
        tolerance: matching radius (arcseconds)
    
    Returns:
        indices of matches in catalog 2 for each source in catalog 1
        (-1 if no match)
    """
    tolerance_deg = tolerance / 3600.0  # Convert arcsec to degrees
    matches = []
    
    for i in range(len(ra1)):
        # Calculate angular separation (simplified for small angles)
        sep = np.sqrt((ra1[i] - ra2)**2 + (dec1[i] - dec2)**2)
        
        # Find closest match
        closest = np.argmin(sep)
        if sep[closest] < tolerance_deg:
            matches.append(closest)
        else:
            matches.append(-1)  # No match
    
    return np.array(matches)
```

This is simplified - proper angular separation on a sphere uses the haversine formula. For production work, use `astropy.coordinates`.

## Combining Data from Multiple Epochs

Compare flux measurements over time to find variable sources:

```python
import numpy as np

# Epoch 1 and 2 measurements
flux_epoch1 = np.array([245.7, 189.3, 312.5, 198.7, 267.4])
flux_epoch2 = np.array([250.2, 191.1, 305.8, 203.1, 268.9])
flux_errors = np.array([10.0, 9.5, 12.0, 8.5, 11.0])

# Calculate fractional change
frac_change = (flux_epoch2 - flux_epoch1) / flux_epoch1

# Is change significant? (> 3-sigma)
significance = np.abs(flux_epoch2 - flux_epoch1) / flux_errors
is_variable = significance > 3

print(f"Variable sources: {np.sum(is_variable)}")
print(f"Indices: {np.where(is_variable)[0]}")
```

## Working with Time Series

Astronomical time series often have irregular sampling and gaps:

```python
import numpy as np
import matplotlib.pyplot as plt

# Load time series (MJD and flux)
data = np.loadtxt('lightcurve.txt')
time = data[:, 0]  # Modified Julian Date
flux = data[:, 1]
errors = data[:, 2]

# Check for gaps
time_diff = np.diff(time)  # Differences between consecutive times
median_cadence = np.median(time_diff)
gaps = time_diff > 5 * median_cadence

print(f"Found {np.sum(gaps)} gaps in time series")

# Plot
plt.errorbar(time, flux, yerr=errors, fmt='o')
plt.xlabel('MJD')
plt.ylabel('Flux (mJy)')
plt.title('Light Curve')
plt.show()
```

## Quality Flags and Data Validation

Catalogs often include quality flags. Use them to filter data:

```python
import numpy as np

# Load catalog with quality flags
data = np.genfromtxt('catalog.csv', delimiter=',', names=True)

ra = data['ra']
dec = data['dec']
flux = data['flux']
quality = data['quality_flag']

# Quality flag meanings: 0=good, 1=marginal, 2=bad
good_data = quality == 0
marginal_data = quality == 1
bad_data = quality == 2

print(f"Good: {np.sum(good_data)}")
print(f"Marginal: {np.sum(marginal_data)}")
print(f"Bad: {np.sum(bad_data)}")

# Work only with good data
flux_clean = flux[good_data]
ra_clean = ra[good_data]
dec_clean = dec[good_data]
```

## Reading FITS Files (Brief Introduction)

FITS is astronomy's standard image/table format. Using `astropy.io.fits`:

```python
from astropy.io import fits

# Open FITS file
with fits.open('observation.fits') as hdul:
    # Print structure
    hdul.info()
    
    # Access data from first extension
    data = hdul[0].data
    header = hdul[0].header
    
    print(f"Image shape: {data.shape}")
    print(f"Observation date: {header['DATE-OBS']}")
```

We won't go deep into FITS here, but this shows the basic pattern. FITS files have headers (metadata) and data (images or tables).

## Things Worth Noting

**Always validate your data after loading.** Check for reasonable values, expected ranges, proper units. A quick histogram or scatter plot catches many issues:

```python
import matplotlib.pyplot as plt
plt.hist(flux, bins=30)
plt.show()  # Look for unexpected spikes or distributions
```

**NaN propagates through calculations.** If you have NaNs and don't handle them, results become NaN:

```python
data = np.array([1, 2, np.nan, 4])
mean = np.mean(data)  # Result: nan

# Use nanmean instead
mean = np.nanmean(data)  # Result: 2.33... (ignores NaN)
```

**Coordinate matching is tricky.** Simple Euclidean distance fails near poles or when RA wraps around 360°→0°. For real work, use `astropy.coordinates.match_coordinates_sky`.

**Time zones and formats matter.** MJD, JD, ISO dates - make sure you know which format your data uses. Mixing them creates errors.

**Memory limits with huge files.** If a file is gigabytes, loading it all at once might fail. Process in chunks or use memory-mapped arrays.

## Try This

1. Load a CSV file with missing values and count how many NaNs are in each column
2. Create synthetic data with outliers, then clean it using both sigma-clipping and IQR methods
3. Generate two "catalogs" with some overlapping sources and cross-match them
4. Simulate a time series with gaps and identify where the gaps are
5. Create a quality flag array and filter data to keep only "good" measurements

## How This Is Typically Used in Astronomy

Every analysis pipeline handles missing data, removes outliers, cross-matches catalogs (LoTSS with FIRST with NVSS), combines multi-epoch observations, applies quality cuts, and reads FITS files.

These practical skills separate working code from robust, production-ready analysis.

## Related Lessons

**Previous**: [19_scipy_essentials.md](19_scipy_essentials.md) - Scientific algorithms

**Next**: [21_best_practices.md](21_best_practices.md) - Writing maintainable code

**Essential tools**: [14_numpy_basics.md](14_numpy_basics.md), [15_numpy_data_loading.md](15_numpy_data_loading.md)
