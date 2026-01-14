# Matplotlib Advanced

Basic line and scatter plots get you far, but astronomy needs more: histograms to see distributions, error bars for uncertainties, contour plots for 2D data, and heatmaps for images. Let's explore these more advanced visualizations.

## Histograms - Understanding Distributions

A histogram shows how many values fall in each range (bin). Essential for understanding your data distribution:

```python
import matplotlib.pyplot as plt
import numpy as np

# Load flux measurements for many sources
flux = np.array([245.7, 189.3, 312.5, 198.7, 267.4, 156.2, 289.3, 
                 223.1, 178.9, 334.2, 201.5, 298.7, 187.6, 256.3])

plt.hist(flux, bins=5, edgecolor='black')
plt.xlabel("Flux (mJy)")
plt.ylabel("Number of Sources")
plt.title("Flux Distribution")
plt.show()
```

The `bins=5` creates 5 bins. Too few bins and you lose detail, too many and you get noise. For N data points, start with √N bins as a rule of thumb.

The `edgecolor='black'` draws lines between bars, making them easier to distinguish.

You can customize further:

```python
plt.hist(flux, bins=10, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel("Flux (mJy)")
plt.ylabel("Count")
plt.grid(True, alpha=0.3)
plt.show()
```

The `alpha=0.7` makes bars semi-transparent (useful when overlaying multiple histograms).

## Multiple Histograms for Comparison

Compare distributions by overlaying histograms:

```python
import matplotlib.pyplot as plt
import numpy as np

catalog_A_flux = np.array([245.7, 189.3, 312.5, 198.7, 267.4])
catalog_B_flux = np.array([156.2, 289.3, 223.1, 178.9, 334.2])

plt.hist(catalog_A_flux, bins=5, alpha=0.5, label='Catalog A', color='blue')
plt.hist(catalog_B_flux, bins=5, alpha=0.5, label='Catalog B', color='red')

plt.xlabel("Flux (mJy)")
plt.ylabel("Count")
plt.legend()
plt.show()
```

The transparency (`alpha`) lets you see overlapping regions. This quickly reveals if two populations have different distributions.

## Error Bars - Showing Uncertainty

Measurements have uncertainties. Error bars communicate this:

```python
import matplotlib.pyplot as plt
import numpy as np

epochs = np.array([1, 2, 3, 4, 5])
flux = np.array([245.7, 238.9, 251.3, 243.2, 247.8])
flux_err = np.array([10.2, 12.5, 9.8, 11.3, 10.7])  # Uncertainties

plt.errorbar(epochs, flux, yerr=flux_err, 
             marker='o', capsize=5, capthick=2,
             linestyle='-', linewidth=1)

plt.xlabel("Epoch")
plt.ylabel("Flux (mJy)")
plt.title("Light Curve with Uncertainties")
plt.show()
```

The `yerr` parameter adds vertical error bars. The `capsize=5` adds caps on the error bars (making them look like ⊢⊣), and `capthick` controls cap line thickness.

For errors in both x and y:

```python
plt.errorbar(x, y, xerr=x_err, yerr=y_err, fmt='o')
```

The `fmt='o'` specifies marker style without connecting lines.

## Logarithmic Histograms

When data spans orders of magnitude, use log bins:

```python
import matplotlib.pyplot as plt
import numpy as np

# Luminosities spanning huge range
luminosity = np.array([1e22, 5e22, 2e23, 8e23, 3e24, 1e25, 4e25])

plt.hist(luminosity, bins=np.logspace(22, 26, 10), edgecolor='black')
plt.xscale('log')
plt.xlabel("Luminosity (W/Hz)")
plt.ylabel("Count")
plt.show()
```

The `np.logspace(22, 26, 10)` creates 10 bins evenly spaced in log space from 10^22 to 10^26. This prevents all your data from cramming into one or two bins.

## Contour Plots - 2D Distributions

For 2D data like density maps or images:

```python
import matplotlib.pyplot as plt
import numpy as np

# Create 2D grid
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

# Some function (e.g., Gaussian)
Z = np.exp(-(X**2 + Y**2) / 10)

plt.contour(X, Y, Z, levels=10, cmap='viridis')
plt.colorbar(label='Intensity')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("2D Distribution")
plt.show()
```

Contour lines connect points of equal value, like elevation on a topographic map. The `levels=10` creates 10 contour lines. The `cmap='viridis'` sets the color scheme.

For filled contours (looks smoother):

```python
plt.contourf(X, Y, Z, levels=20, cmap='viridis')
plt.colorbar()
plt.show()
```

## Heatmaps - Showing 2D Arrays

Display a 2D array as a color-coded image:

```python
import matplotlib.pyplot as plt
import numpy as np

# Simulated radio image (10x10 pixels)
image = np.random.rand(10, 10) * 100

plt.imshow(image, cmap='hot', origin='lower')
plt.colorbar(label='Flux (mJy/beam)')
plt.xlabel("X pixel")
plt.ylabel("Y pixel")
plt.title("Radio Source")
plt.show()
```

The `origin='lower'` puts (0,0) at the bottom-left (standard for astronomy). Default is top-left (standard for computer graphics).

Common colormaps for astronomy:
- `'hot'` - black → red → yellow → white (good for intensity)
- `'viridis'` - perceptually uniform, colorblind-friendly
- `'gray'` - grayscale
- `'RdBu_r'` - red-blue diverging (good for showing positive/negative)

## Box Plots - Comparing Distributions

Box plots show median, quartiles, and outliers compactly:

```python
import matplotlib.pyplot as plt
import numpy as np

catalog_A = np.array([245.7, 189.3, 312.5, 198.7, 267.4, 234.1, 289.6])
catalog_B = np.array([156.2, 289.3, 223.1, 178.9, 334.2, 201.5, 298.7])

data = [catalog_A, catalog_B]
plt.boxplot(data, labels=['Catalog A', 'Catalog B'])
plt.ylabel("Flux (mJy)")
plt.title("Flux Distribution Comparison")
plt.grid(True, alpha=0.3)
plt.show()
```

The box shows the interquartile range (25th to 75th percentile), the line inside is the median, the whiskers extend to 1.5×IQR, and points beyond are potential outliers.

## Things Worth Noting

**Choosing bin numbers for histograms:** Too few bins hide structure, too many show noise. Rules of thumb:
- Sturges' rule: bins = log₂(N) + 1
- Square root rule: bins = √N
- Or just experiment with a few values

**Error bars in log space:** If you have log-scale axes with error bars, the errors should be in the same space as the data. For symmetric errors on log plots, use:

```python
plt.errorbar(x, y, yerr=[y_lower_err, y_upper_err], ...)
```

where errors are asymmetric since log space isn't linear.

**Colorbars:** Always add a colorbar to heatmaps and contour plots with `plt.colorbar()`. Otherwise readers don't know what the colors mean.

**Figure size for subplots:** When making many panels, increase figure size proportionally: `figsize=(12, 8)` for a 2×2 grid.

**Saving with transparent backgrounds:**

```python
plt.savefig('plot.png', transparent=True, dpi=300)
```

Useful for overlaying plots on slides.

## Try This

1. Create a histogram of 1000 random flux values with appropriate bin number
2. Make an errorbar plot with both x and y uncertainties
3. Create a 2×1 subplot: histogram on top, boxplot below, both showing same data
4. Generate a heatmap from a 20×20 array of random values
5. Make overlapping histograms for two catalogs with different colors and transparency

## How This Is Typically Used in Astronomy

Histograms for luminosity functions and redshift distributions, error bars on light curves and spectral measurements, contour plots for radio source structure, heatmaps for survey fields or CCD images, and box plots for comparing populations.

These visualization types appear constantly in astronomy papers.

## Related Lessons

**Previous**: [16_matplotlib_basics.md](16_matplotlib_basics.md) - Basic line and scatter plots

**Next**: [18_astronomy_calculations.md](18_astronomy_calculations.md) - Domain-specific calculations

**Uses**: [14_numpy_basics.md](14_numpy_basics.md) - NumPy arrays for the data
