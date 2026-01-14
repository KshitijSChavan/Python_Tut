# Answers for Try This Section in 17 - Matplotlib Advanced

## Question 1
**Original question:** Create a histogram of 1000 random flux values with appropriate bin number

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate 1000 random flux values (normal distribution)
np.random.seed(42)  # For reproducibility
flux = np.random.normal(loc=200, scale=50, size=1000)

# Use square root rule for bins: sqrt(N)
n_bins = int(np.sqrt(len(flux)))
print(f"Using {n_bins} bins for {len(flux)} values")

plt.figure(figsize=(10, 6))
plt.hist(flux, bins=n_bins, edgecolor='black', alpha=0.7)
plt.xlabel('Flux (mJy)')
plt.ylabel('Count')
plt.title(f'Flux Distribution (N={len(flux)}, bins={n_bins})')
plt.grid(True, alpha=0.3)
plt.show()
```

## Question 2
**Original question:** Make an errorbar plot with both x and y uncertainties

```python
import matplotlib.pyplot as plt
import numpy as np

freq = np.array([144, 323, 608, 1400])
flux = np.array([245.7, 189.3, 156.2, 98.5])
flux_err = np.array([10.2, 12.5, 9.8, 8.3])
freq_err = np.array([5, 10, 15, 20])  # Frequency uncertainties

plt.errorbar(freq, flux, xerr=freq_err, yerr=flux_err, 
             fmt='o', capsize=5, capthick=2, 
             markersize=8, linewidth=2)
plt.xlabel('Frequency (MHz)')
plt.ylabel('Flux (mJy)')
plt.title('Spectral Measurements with Uncertainties')
plt.grid(True, alpha=0.3)
plt.show()
```

## Question 3
**Original question:** Create a 2×1 subplot: histogram on top, boxplot below, both showing same data

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
np.random.seed(42)
flux = np.random.normal(loc=250, scale=50, size=100)

# Create 2x1 subplot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Top: Histogram
ax1.hist(flux, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
ax1.set_ylabel('Count')
ax1.set_title('Flux Distribution')
ax1.grid(True, alpha=0.3)

# Bottom: Boxplot
ax2.boxplot(flux, vert=False, widths=0.5)
ax2.set_xlabel('Flux (mJy)')
ax2.set_title('Flux Statistics (Boxplot)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print statistics
print(f"Mean: {np.mean(flux):.2f} mJy")
print(f"Median: {np.median(flux):.2f} mJy")
print(f"Q1: {np.percentile(flux, 25):.2f} mJy")
print(f"Q3: {np.percentile(flux, 75):.2f} mJy")
```

## Question 4
**Original question:** Generate a heatmap from a 20×20 array of random values

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate 20x20 random data
np.random.seed(42)
image_data = np.random.rand(20, 20) * 100  # Random values 0-100

plt.figure(figsize=(10, 8))
plt.imshow(image_data, cmap='hot', origin='lower')
plt.colorbar(label='Flux (mJy/beam)')
plt.xlabel('X pixel')
plt.ylabel('Y pixel')
plt.title('Simulated Radio Image (20×20)')

# Add grid for clarity
plt.grid(False)  # Turn off default grid
for i in range(21):
    plt.axhline(i-0.5, color='gray', linewidth=0.5, alpha=0.3)
    plt.axvline(i-0.5, color='gray', linewidth=0.5, alpha=0.3)

plt.show()

print(f"Data shape: {image_data.shape}")
print(f"Min value: {np.min(image_data):.2f}")
print(f"Max value: {np.max(image_data):.2f}")
print(f"Mean value: {np.mean(image_data):.2f}")
```

## Question 5
**Original question:** Make overlapping histograms for two catalogs with different colors and transparency

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate two catalogs with different distributions
np.random.seed(42)
catalog_A = np.random.normal(loc=200, scale=30, size=500)
catalog_B = np.random.normal(loc=250, scale=40, size=500)

plt.figure(figsize=(10, 6))

# Overlapping histograms
plt.hist(catalog_A, bins=30, alpha=0.5, label='Catalog A', 
         color='blue', edgecolor='black')
plt.hist(catalog_B, bins=30, alpha=0.5, label='Catalog B', 
         color='red', edgecolor='black')

plt.xlabel('Flux (mJy)')
plt.ylabel('Count')
plt.title('Comparing Two Catalogs')
plt.legend()
plt.grid(True, alpha=0.3)

# Add vertical lines for means
mean_A = np.mean(catalog_A)
mean_B = np.mean(catalog_B)
plt.axvline(mean_A, color='blue', linestyle='--', linewidth=2, 
            label=f'Mean A: {mean_A:.1f}')
plt.axvline(mean_B, color='red', linestyle='--', linewidth=2, 
            label=f'Mean B: {mean_B:.1f}')

plt.legend()
plt.show()

print(f"Catalog A: mean={mean_A:.2f}, std={np.std(catalog_A):.2f}")
print(f"Catalog B: mean={mean_B:.2f}, std={np.std(catalog_B):.2f}")
```
