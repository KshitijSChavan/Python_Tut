# Answers for Try This Section in 16 - Matplotlib Basics

## Question 1
**Original question:** Create a line plot of flux vs frequency with logarithmic x-axis

```python
import matplotlib.pyplot as plt
import numpy as np

freq = np.array([144, 323, 608, 1400, 4850])
flux = np.array([245.7, 189.3, 156.2, 98.5, 45.2])

plt.plot(freq, flux, marker='o')
plt.xscale('log')  # Logarithmic x-axis
plt.xlabel('Frequency (MHz)')
plt.ylabel('Flux (mJy)')
plt.title('Spectral Energy Distribution')
plt.grid(True, alpha=0.3)
plt.show()
```

## Question 2
**Original question:** Make a scatter plot comparing RA vs Dec for several sources

```python
import matplotlib.pyplot as plt

# Source positions
ra = [187.7, 221.2, 149.5, 235.8, 201.3]
dec = [12.3, 31.5, 32.4, 15.5, 28.7]

plt.scatter(ra, dec, s=100, alpha=0.7)
plt.xlabel('RA (degrees)')
plt.ylabel('Dec (degrees)')
plt.title('Source Positions')
plt.grid(True, alpha=0.3)

# Add source labels
for i, (r, d) in enumerate(zip(ra, dec)):
    plt.annotate(f'S{i+1}', (r, d), xytext=(5, 5), 
                textcoords='offset points', fontsize=9)

plt.show()
```

## Question 3
**Original question:** Plot two datasets on the same axes with different colors and a legend

```python
import matplotlib.pyplot as plt
import numpy as np

epochs = np.array([1, 2, 3, 4, 5])
source_A = np.array([245.7, 238.9, 251.3, 243.2, 247.8])
source_B = np.array([189.3, 195.2, 187.8, 192.1, 190.5])

plt.plot(epochs, source_A, marker='o', color='blue', 
         label='Source A', linewidth=2)
plt.plot(epochs, source_B, marker='s', color='red', 
         label='Source B', linewidth=2)

plt.xlabel('Epoch')
plt.ylabel('Flux (mJy)')
plt.title('Comparing Two Sources')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Question 4
**Original question:** Create a 2x2 subplot grid with four different measurements

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
epochs = np.array([1, 2, 3, 4, 5])
flux = np.array([245.7, 238.9, 251.3, 243.2, 247.8])
size = np.array([45.2, 44.8, 46.1, 45.5, 45.9])
spectral_index = np.array([-0.75, -0.72, -0.78, -0.74, -0.76])
redshift = np.array([1.42, 1.41, 1.43, 1.42, 1.42])

# Create 2x2 subplot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))

# Top left: Flux
ax1.plot(epochs, flux, marker='o', color='blue')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Flux (mJy)')
ax1.set_title('Flux vs Time')
ax1.grid(True, alpha=0.3)

# Top right: Size
ax2.plot(epochs, size, marker='s', color='green')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Size (arcsec)')
ax2.set_title('Angular Size')
ax2.grid(True, alpha=0.3)

# Bottom left: Spectral Index
ax3.plot(epochs, spectral_index, marker='^', color='red')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Spectral Index')
ax3.set_title('Spectral Index')
ax3.grid(True, alpha=0.3)

# Bottom right: Redshift
ax4.plot(epochs, redshift, marker='d', color='purple')
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Redshift')
ax4.set_title('Redshift')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Question 5
**Original question:** Save a plot as both PNG and PDF with high resolution

```python
import matplotlib.pyplot as plt
import numpy as np

freq = np.array([144, 323, 608, 1400])
flux = np.array([245.7, 189.3, 156.2, 98.5])

plt.figure(figsize=(8, 6))
plt.scatter(freq, flux, s=100)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Flux (mJy)')
plt.title('Spectral Energy Distribution')
plt.grid(True, alpha=0.3)

# Save as PNG with high resolution
plt.savefig('sed_plot.png', dpi=300, bbox_inches='tight')
print("Saved as sed_plot.png")

# Save as PDF (vector format)
plt.savefig('sed_plot.pdf', bbox_inches='tight')
print("Saved as sed_plot.pdf")

plt.show()
```
