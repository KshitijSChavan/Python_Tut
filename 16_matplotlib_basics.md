# Matplotlib Basics

Numbers in tables are useful, but visualizing data reveals patterns you'd never see otherwise. **Matplotlib** is Python's main plotting library - it can create publication-quality figures for papers, quick plots for exploration, or complex multi-panel visualizations.

## Do You Have Matplotlib?

Check if matplotlib is installed:

```bash
python3 -c "import matplotlib; print(matplotlib.__version__)"
```

If you see a version number, you're ready. If not, install it:

```bash
pip3 install matplotlib --break-system-packages
```

## Your First Plot

The simplest plot - flux measurements over time:

```python
import matplotlib.pyplot as plt

epochs = [1, 2, 3, 4, 5]
flux = [245.7, 238.9, 251.3, 243.2, 247.8]

plt.plot(epochs, flux)
plt.xlabel("Epoch")
plt.ylabel("Flux (mJy)")
plt.title("Source Variability")
plt.show()
```

The `plt.show()` command opens a window with your plot. In Jupyter notebooks, plots appear automatically below the cell.

What's happening here? `plt.plot(x, y)` creates a line connecting your points. The `xlabel`, `ylabel`, and `title` functions add labels. Everything builds up until `show()` displays it.

## Line Plots vs Scatter Plots

Line plots connect points with lines - good for continuous data like time series:

```python
import matplotlib.pyplot as plt

time = [0, 1, 2, 3, 4, 5]
flux = [245.7, 238.9, 251.3, 243.2, 247.8, 249.1]

plt.plot(time, flux, marker='o')  # Line with markers at points
plt.xlabel("Time (days)")
plt.ylabel("Flux (mJy)")
plt.show()
```

Scatter plots show individual points without connecting lines - good for comparing two unrelated quantities:

```python
import matplotlib.pyplot as plt

freq = [144, 323, 608, 1400]
flux = [245.7, 189.3, 156.2, 98.5]

plt.scatter(freq, flux)
plt.xlabel("Frequency (MHz)")
plt.ylabel("Flux (mJy)")
plt.title("Spectral Energy Distribution")
plt.show()
```

When would you use each? Time series, temperature curves, light curves → line plot. Flux vs frequency for different sources, RA vs Dec positions, any two properties where order doesn't matter → scatter plot.

## Customizing Appearance

Make your plots clearer with colors, markers, and line styles:

```python
import matplotlib.pyplot as plt

epochs = [1, 2, 3, 4, 5]
flux = [245.7, 238.9, 251.3, 243.2, 247.8]

plt.plot(epochs, flux, 
         color='blue',        # Line color
         marker='o',          # Point marker
         linestyle='--',      # Dashed line
         linewidth=2,         # Thicker line
         markersize=8,        # Larger markers
         label='Source A')    # For legend

plt.xlabel("Epoch")
plt.ylabel("Flux (mJy)")
plt.legend()  # Shows the label
plt.grid(True)  # Add grid lines
plt.show()
```

Common colors: `'blue'`, `'red'`, `'green'`, `'black'`, `'orange'`. Or use shortcuts: `'b'`, `'r'`, `'g'`, `'k'`, `'orange'`.

Common markers: `'o'` (circle), `'s'` (square), `'^'` (triangle), `'*'` (star), `'+'` (plus).

## Multiple Lines on One Plot

Compare different sources or epochs:

```python
import matplotlib.pyplot as plt

epochs = [1, 2, 3, 4, 5]
source_A = [245.7, 238.9, 251.3, 243.2, 247.8]
source_B = [189.3, 195.2, 187.8, 192.1, 190.5]

plt.plot(epochs, source_A, marker='o', label='Source A')
plt.plot(epochs, source_B, marker='s', label='Source B')

plt.xlabel("Epoch")
plt.ylabel("Flux (mJy)")
plt.title("Comparing Two Sources")
plt.legend()
plt.show()
```

Each `plt.plot()` call adds another line. The `label` parameter combined with `plt.legend()` creates a legend automatically.

## Logarithmic Scales

Astronomy often needs log scales - spectral indices, luminosity functions, anything spanning orders of magnitude:

```python
import matplotlib.pyplot as plt

freq = [144, 323, 608, 1400, 4850]
flux = [245.7, 189.3, 156.2, 98.5, 45.2]

plt.scatter(freq, flux)
plt.xscale('log')  # Log scale on x-axis
plt.yscale('log')  # Log scale on y-axis
plt.xlabel("Frequency (MHz)")
plt.ylabel("Flux (mJy)")
plt.title("Spectral Index (log-log)")
plt.grid(True, which='both', alpha=0.3)
plt.show()
```

On a log-log plot, power laws become straight lines. A spectral index of α = -0.7 means flux ∝ frequency^(-0.7), which is a straight line on log-log axes.

## Saving Figures

For papers or presentations, save plots instead of just viewing them:

```python
import matplotlib.pyplot as plt

epochs = [1, 2, 3, 4, 5]
flux = [245.7, 238.9, 251.3, 243.2, 247.8]

plt.plot(epochs, flux, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Flux (mJy)")

plt.savefig("variability_plot.png", dpi=300, bbox_inches='tight')
plt.show()
```

The `dpi=300` gives high resolution (300 dots per inch - publication quality). The `bbox_inches='tight'` removes extra whitespace. Common formats: `.png` (for presentations), `.pdf` (for papers), `.jpg` (for web).

Save before `show()`, or the figure might be empty after showing.

## Subplots - Multiple Panels

Create multi-panel figures for comparing datasets:

```python
import matplotlib.pyplot as plt

epochs = [1, 2, 3, 4, 5]
flux_A = [245.7, 238.9, 251.3, 243.2, 247.8]
flux_B = [189.3, 195.2, 187.8, 192.1, 190.5]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))  # 1 row, 2 columns

# Left panel
ax1.plot(epochs, flux_A, marker='o')
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Flux (mJy)")
ax1.set_title("Source A")

# Right panel
ax2.plot(epochs, flux_B, marker='s', color='red')
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Flux (mJy)")
ax2.set_title("Source B")

plt.tight_layout()  # Prevents labels from overlapping
plt.show()
```

Notice we use `ax1.plot()` instead of `plt.plot()`. With subplots, you work on individual axes objects. The `figsize=(10, 4)` controls overall figure size in inches.

## Things Worth Noting

**Figure won't show?** Make sure you call `plt.show()` at the end. In scripts, nothing appears without it. In Jupyter notebooks, it's sometimes automatic.

**Plot looks crowded?** Adjust figure size: `plt.figure(figsize=(10, 6))` before plotting.

**Labels cut off?** Use `plt.tight_layout()` before `show()` or `savefig()`.

**Want to clear and start fresh?**

```python
plt.clf()  # Clear current figure
plt.close()  # Close current figure window
```

**Matplotlib uses NumPy arrays efficiently.** If you pass lists, matplotlib converts them internally. For large datasets, use NumPy arrays directly for speed.

**Colors and styles:** You can use hex codes for exact colors: `color='#FF5733'`. Or use built-in colormaps for gradients.

## Try This

1. Create a line plot of flux vs frequency with logarithmic x-axis
2. Make a scatter plot comparing RA vs Dec for several sources
3. Plot two datasets on the same axes with different colors and a legend
4. Create a 2x2 subplot grid with four different measurements
5. Save a plot as both PNG and PDF with high resolution

## How This Is Typically Used in Astronomy

Creating light curves (flux vs time), spectral energy distributions (flux vs frequency), color-magnitude diagrams, sky position plots (RA vs Dec), quality control visualizations, and figures for papers and presentations.

Every astronomy paper has matplotlib figures - it's the standard.

## Related Lessons

**Previous**: [15_numpy_data_loading.md](15_numpy_data_loading.md) - Loading the data to plot

**Next**: [17_matplotlib_advanced.md](17_matplotlib_advanced.md) - Histograms, error bars, more complex plots

**Works best with**: [14_numpy_basics.md](14_numpy_basics.md) - NumPy arrays for large datasets
