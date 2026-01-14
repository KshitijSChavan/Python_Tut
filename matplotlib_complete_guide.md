# Matplotlib - Complete Guide

## Introduction

Matplotlib is Python's most popular plotting library for creating static, animated, and interactive visualizations. It provides a MATLAB-like interface and gives you complete control over every aspect of your plots - from colors and markers to axis labels and figure sizes.

**Key Capabilities:**
- Line plots, scatter plots, bar charts, histograms
- Subplots and multi-panel figures
- Customizable colors, markers, line styles
- Annotations, legends, and labels
- Saving plots as images (PNG, PDF, SVG)
- 2D and basic 3D plotting

**Installation:**
```bash
pip install matplotlib
```

---

## Example 1: Basic Line Plot with Multiple Lines

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.sin(x) * np.exp(-x/10)

# Create figure and plot
plt.figure(figsize=(10, 6))
plt.plot(x, y1, color='blue', linewidth=2, linestyle='-', label='sin(x)', marker='o', markevery=10)
plt.plot(x, y2, color='red', linewidth=2, linestyle='--', label='cos(x)')
plt.plot(x, y3, color='green', linewidth=2, linestyle=':', label='damped sin(x)')

# Customize plot
plt.xlabel('X-axis', fontsize=12)
plt.ylabel('Y-axis', fontsize=12)
plt.title('Multiple Functions Plotted Together', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right', fontsize=10)
plt.xlim(0, 10)
plt.ylim(-1.5, 1.5)

# Save and display
plt.savefig('multiple_lines.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Code Explanation:

**Line 1-2:** Import matplotlib's pyplot module (standard convention is `plt`) and numpy for generating data.

**Line 5-7:** Create data arrays using NumPy. `np.linspace(0, 10, 100)` generates 100 evenly spaced points between 0 and 10. We create three different mathematical functions to demonstrate multiple lines on one plot.

**Line 10:** `plt.figure(figsize=(10, 6))` creates a new figure with width=10 inches and height=6 inches. This must be called before plotting if you want to set figure size.

**Line 11:** `plt.plot()` is the main plotting function. Parameters:
- `x, y1` - data to plot (x-coordinates and y-coordinates)
- `color='blue'` - line color (can use names like 'red', 'green' or hex codes like '#FF5733')
- `linewidth=2` - thickness of the line
- `linestyle='-'` - line style: '-' (solid), '--' (dashed), ':' (dotted), '-.' (dash-dot)
- `label='sin(x)'` - label for legend
- `marker='o'` - adds circular markers at data points
- `markevery=10` - shows markers at every 10th point (not all 100 points)

**Line 12-13:** Two more `plt.plot()` calls add additional lines to the same figure. Each line has different styling.

**Line 16:** `plt.xlabel()` sets the x-axis label. `fontsize` parameter controls text size.

**Line 17:** `plt.ylabel()` sets the y-axis label.

**Line 18:** `plt.title()` adds a title above the plot. `fontweight='bold'` makes it bold.

**Line 19:** `plt.grid(True, alpha=0.3)` adds a grid to the plot. `alpha=0.3` makes it 30% opaque (70% transparent) so it doesn't dominate.

**Line 20:** `plt.legend()` displays the legend box. `loc='upper right'` positions it in the upper right corner. Other options: 'upper left', 'lower left', 'lower right', 'center', 'best' (auto-positions to avoid data).

**Line 21-22:** `plt.xlim()` and `plt.ylim()` set the axis limits explicitly. Without these, matplotlib auto-scales based on data.

**Line 25:** `plt.savefig()` saves the plot to a file. Parameters:
- First argument is filename (extension determines format: .png, .pdf, .svg, .jpg)
- `dpi=300` sets resolution (dots per inch) - 300 is publication quality
- `bbox_inches='tight'` removes extra white space around the figure

**Line 26:** `plt.show()` displays the plot in a window (or inline in Jupyter notebooks).

---

## Example 2: Scatter Plot with Color Mapping

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate random data
np.random.seed(42)
n_points = 200
x = np.random.randn(n_points)
y = np.random.randn(n_points)
colors = x + y  # Color based on sum of x and y
sizes = np.random.randint(20, 200, n_points)  # Random sizes

# Create scatter plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(x, y, c=colors, s=sizes, alpha=0.6, 
                     cmap='viridis', edgecolors='black', linewidth=0.5)

# Add colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('X + Y value', fontsize=12)

# Customize
plt.xlabel('X values', fontsize=12)
plt.ylabel('Y values', fontsize=12)
plt.title('Scatter Plot with Color Mapping and Variable Sizes', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')
plt.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
plt.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.5)

# Save and display
plt.tight_layout()
plt.savefig('scatter_colormap.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Code Explanation:

**Line 5:** `np.random.seed(42)` sets the random seed for reproducibility - same "random" numbers every time.

**Line 6-10:** Generate synthetic data:
- `n_points = 200` - number of data points
- `np.random.randn()` generates random numbers from a standard normal distribution (mean=0, std=1)
- `colors = x + y` - creates color values based on data (higher x+y = different color)
- `np.random.randint(20, 200, n_points)` generates random integers between 20 and 200 for marker sizes

**Line 14:** `plt.scatter()` creates a scatter plot. Key parameters:
- `x, y` - coordinates of points
- `c=colors` - color values for each point (can be array for color mapping or single color like 'red')
- `s=sizes` - size of each marker (can be single value or array for variable sizes)
- `alpha=0.6` - transparency (0=invisible, 1=opaque)
- `cmap='viridis'` - colormap name (other popular ones: 'plasma', 'inferno', 'coolwarm', 'RdYlBu')
- `edgecolors='black'` - border color around each marker
- `linewidth=0.5` - thickness of marker borders

**Line 18:** `plt.colorbar(scatter)` adds a colorbar showing the color scale. Must pass the scatter plot object.

**Line 19:** `cbar.set_label()` adds a label to the colorbar explaining what the colors represent.

**Line 26-27:** `plt.axhline()` and `plt.axvline()` add horizontal and vertical reference lines:
- `y=0` or `x=0` - position of the line
- `color='red'` - line color
- `linestyle='--'` - dashed line
- `alpha=0.5` - semi-transparent

**Line 30:** `plt.tight_layout()` automatically adjusts subplot parameters to give specified padding and prevent labels from overlapping or being cut off.

---

## Example 3: Subplots - Multiple Plots in One Figure

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.tan(x)
y4 = x**2

# Create figure with 2x2 subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Line plot
ax1.plot(x, y1, 'b-', linewidth=2)
ax1.set_title('Sine Wave', fontsize=12, fontweight='bold')
ax1.set_xlabel('X')
ax1.set_ylabel('sin(x)')
ax1.grid(True, alpha=0.3)

# Plot 2: Line plot with filled area
ax2.plot(x, y2, 'r-', linewidth=2)
ax2.fill_between(x, y2, 0, alpha=0.3, color='red')
ax2.set_title('Cosine Wave with Fill', fontsize=12, fontweight='bold')
ax2.set_xlabel('X')
ax2.set_ylabel('cos(x)')
ax2.grid(True, alpha=0.3)

# Plot 3: Limited range (dealing with discontinuities)
y3_clipped = np.clip(y3, -10, 10)
ax3.plot(x, y3_clipped, 'g-', linewidth=2)
ax3.set_title('Tangent (clipped)', fontsize=12, fontweight='bold')
ax3.set_xlabel('X')
ax3.set_ylabel('tan(x)')
ax3.set_ylim(-10, 10)
ax3.grid(True, alpha=0.3)

# Plot 4: Scatter with color
ax4.scatter(x, y4, c=y4, cmap='plasma', s=50)
ax4.set_title('Quadratic Function', fontsize=12, fontweight='bold')
ax4.set_xlabel('X')
ax4.set_ylabel('x²')
ax4.grid(True, alpha=0.3)

# Overall title
fig.suptitle('Multiple Subplots Demonstration', fontsize=16, fontweight='bold', y=0.995)

# Adjust spacing between subplots
plt.tight_layout()

# Save and display
plt.savefig('subplots_demo.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Code Explanation:

**Line 11:** `plt.subplots(2, 2, figsize=(12, 10))` creates a figure with 2x2 grid of subplots:
- First argument: number of rows (2)
- Second argument: number of columns (2)
- Returns: `fig` (figure object) and `axes` (array of subplot axes)
- Unpacking `((ax1, ax2), (ax3, ax4))` gives us individual names for each subplot

This is different from creating separate figures - all subplots share one figure object.

**Line 14:** `ax1.plot()` - Notice we're calling `plot()` on the specific axes object (`ax1`), not on `plt`. This is the "object-oriented" style of matplotlib, which is better for complex figures with multiple subplots.

**Line 15:** `ax1.set_title()` sets the title for this specific subplot (not the whole figure).

**Line 16-17:** `ax1.set_xlabel()` and `ax1.set_ylabel()` - same as `plt.xlabel()` but for specific subplot.

**Line 22:** `ax2.fill_between(x, y2, 0, alpha=0.3, color='red')` fills the area between the curve and y=0:
- `x` - x-coordinates
- `y2` - upper boundary (the curve)
- `0` - lower boundary (can be another array or constant)
- Creates shaded region under/over the curve

**Line 30:** `np.clip(y3, -10, 10)` clips values to range [-10, 10]. Tangent has discontinuities (goes to ±infinity), so we clip it for better visualization.

**Line 35:** `ax3.set_ylim(-10, 10)` explicitly sets y-axis limits for this subplot only.

**Line 38:** Color mapping in scatter plot within subplot works the same as standalone scatter plots.

**Line 45:** `fig.suptitle()` sets a title for the entire figure (super-title), above all subplots. `y=0.995` positions it at the very top.

**Line 48:** `plt.tight_layout()` is especially important for subplots - prevents titles/labels from overlapping between adjacent plots.

---

## Example 4: Histogram and Statistical Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate random data from different distributions
np.random.seed(42)
data_normal = np.random.normal(100, 15, 1000)
data_uniform = np.random.uniform(50, 150, 1000)
data_exponential = np.random.exponential(20, 1000)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histogram 1: Basic histogram
axes[0, 0].hist(data_normal, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Normal Distribution', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Value')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].axvline(np.mean(data_normal), color='red', linestyle='--', 
                    linewidth=2, label=f'Mean: {np.mean(data_normal):.2f}')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Histogram 2: Multiple overlapping histograms
axes[0, 1].hist(data_normal, bins=25, alpha=0.5, label='Normal', color='blue', edgecolor='black')
axes[0, 1].hist(data_uniform, bins=25, alpha=0.5, label='Uniform', color='red', edgecolor='black')
axes[0, 1].set_title('Comparing Distributions', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Value')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Histogram 3: Normalized (probability density)
axes[1, 0].hist(data_exponential, bins=30, density=True, color='green', 
                edgecolor='black', alpha=0.7)
axes[1, 0].set_title('Exponential Distribution (Normalized)', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Value')
axes[1, 0].set_ylabel('Probability Density')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Histogram 4: Cumulative histogram
axes[1, 1].hist(data_normal, bins=30, cumulative=True, color='purple', 
                edgecolor='black', alpha=0.7)
axes[1, 1].set_title('Cumulative Histogram', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Value')
axes[1, 1].set_ylabel('Cumulative Frequency')
axes[1, 1].grid(True, alpha=0.3)

# Overall title
fig.suptitle('Histogram Visualization Techniques', fontsize=16, fontweight='bold')

# Adjust layout and save
plt.tight_layout()
plt.savefig('histograms.png', dpi=300, bbox_inches='tight')
plt.show()

# Print statistics
print(f"Normal - Mean: {np.mean(data_normal):.2f}, Std: {np.std(data_normal):.2f}")
print(f"Uniform - Mean: {np.mean(data_uniform):.2f}, Std: {np.std(data_uniform):.2f}")
print(f"Exponential - Mean: {np.mean(data_exponential):.2f}, Std: {np.std(data_exponential):.2f}")
```

### Code Explanation:

**Line 5-8:** Generate three different types of random data:
- `np.random.normal(100, 15, 1000)` - 1000 samples from normal distribution with mean=100, std=15
- `np.random.uniform(50, 150, 1000)` - 1000 samples uniformly distributed between 50 and 150
- `np.random.exponential(20, 1000)` - 1000 samples from exponential distribution with scale=20

**Line 14:** `axes[0, 0].hist()` creates a histogram. Key parameters:
- First argument: data array
- `bins=30` - number of bins (bars) to divide the data into. Can also be a sequence of bin edges.
- `color='skyblue'` - fill color of bars
- `edgecolor='black'` - color of bar borders
- `alpha=0.7` - transparency

**Line 18-19:** `axes[0, 0].axvline()` adds vertical line at the mean:
- `np.mean(data_normal)` calculates the mean
- `label=f'Mean: {np.mean(data_normal):.2f}'` uses f-string formatting to show value rounded to 2 decimals

**Line 21:** `grid(True, alpha=0.3, axis='y')` - `axis='y'` shows grid only for horizontal lines (y-axis), not vertical

**Line 24-25:** Multiple histograms on same axes:
- Both `hist()` calls target the same subplot `axes[0, 1]`
- `alpha=0.5` makes them semi-transparent so you can see overlap
- Different colors distinguish the distributions
- `label` creates legend entries

**Line 33:** `density=True` normalizes the histogram so that the area under the bars equals 1. This gives probability density instead of counts. Useful for comparing distributions with different sample sizes.

**Line 42:** `cumulative=True` creates a cumulative histogram where each bar shows the count up to that point, not just the count in that bin. The final bar always reaches the total count.

**Line 56-58:** Print statistics using f-strings with `.2f` formatting (2 decimal places).

---

## Example 5: Advanced Customization - Error Bars and Annotations

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate data with measurement errors
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([2.3, 4.1, 5.8, 7.9, 10.2, 12.1, 14.5, 16.3, 18.9, 20.5])
y_error = np.array([0.3, 0.4, 0.3, 0.5, 0.4, 0.6, 0.5, 0.4, 0.6, 0.5])
x_error = np.array([0.1, 0.1, 0.15, 0.1, 0.2, 0.1, 0.15, 0.1, 0.1, 0.2])

# Create figure
fig, ax = plt.subplots(figsize=(12, 8))

# Plot with error bars
ax.errorbar(x, y, yerr=y_error, xerr=x_error, fmt='o', markersize=8, 
            color='blue', ecolor='red', elinewidth=2, capsize=5, capthick=2,
            label='Measured Data')

# Add a best-fit line
coeffs = np.polyfit(x, y, 1)
fit_line = np.poly1d(coeffs)
x_fit = np.linspace(1, 10, 100)
ax.plot(x_fit, fit_line(x_fit), 'g--', linewidth=2, label=f'Fit: y={coeffs[0]:.2f}x+{coeffs[1]:.2f}')

# Add annotations
ax.annotate('Outlier?', xy=(5, 10.2), xytext=(6.5, 8),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=12, color='red', fontweight='bold')

ax.annotate('Linear trend', xy=(7, 14.5), xytext=(3, 17),
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            fontsize=12, color='green')

# Add text box with statistics
textstr = f'Slope: {coeffs[0]:.3f}\nIntercept: {coeffs[1]:.3f}\nR²: {np.corrcoef(x, y)[0,1]**2:.3f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props)

# Customize
ax.set_xlabel('X Variable', fontsize=13, fontweight='bold')
ax.set_ylabel('Y Variable', fontsize=13, fontweight='bold')
ax.set_title('Data with Error Bars and Annotations', fontsize=15, fontweight='bold', pad=20)
ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(0, 11)
ax.set_ylim(0, 23)

# Save and display
plt.tight_layout()
plt.savefig('errorbars_annotations.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Code Explanation:

**Line 5-8:** Create data arrays with measurement uncertainties:
- `y_error` - uncertainty in y-direction (vertical)
- `x_error` - uncertainty in x-direction (horizontal)

**Line 14:** `ax.errorbar()` plots data with error bars. Parameters:
- `x, y` - data coordinates
- `yerr=y_error` - vertical error bar sizes (can be single value or array)
- `xerr=x_error` - horizontal error bar sizes
- `fmt='o'` - format string: 'o' means circles, '-' would add connecting line
- `markersize=8` - size of data point markers
- `color='blue'` - color of data points
- `ecolor='red'` - color of error bars (can be different from marker color)
- `elinewidth=2` - thickness of error bar lines
- `capsize=5` - length of error bar caps (horizontal/vertical lines at ends)
- `capthick=2` - thickness of caps

**Line 19:** `np.polyfit(x, y, 1)` fits a polynomial of degree 1 (linear fit) to the data. Returns coefficients [slope, intercept].

**Line 20:** `np.poly1d(coeffs)` creates a polynomial function from coefficients that we can call like a function.

**Line 21-22:** Generate smooth x values and calculate fitted y values to plot the fit line.

**Line 25-27:** `ax.annotate()` adds annotation with arrow. Parameters:
- `'Outlier?'` - text to display
- `xy=(5, 10.2)` - coordinates of the point being annotated (where arrow points)
- `xytext=(6.5, 8)` - coordinates where text appears
- `arrowprops=dict(...)` - dictionary of arrow properties:
  - `arrowstyle='->'` - arrow style (options: '->', '-', '-[', '|-|', 'fancy', 'wedge')
  - `color='red'` - arrow color
  - `lw=2` - line width

**Line 34-35:** Create a text string with statistics using f-strings. `\n` creates line breaks.

**Line 36:** `dict(boxstyle='round', facecolor='wheat', alpha=0.5)` defines properties for text box:
- `boxstyle='round'` - rounded corners (options: 'square', 'round', 'roundtooth', 'sawtooth')
- `facecolor='wheat'` - background color
- `alpha=0.5` - transparency

**Line 37-38:** `ax.text()` places text box:
- `0.05, 0.95` - coordinates in axes fraction (0-1 range)
- `transform=ax.transAxes` - IMPORTANT: uses axes coordinates (0-1) instead of data coordinates
- `verticalalignment='top'` - aligns text to top of specified position
- `bbox=props` - applies the box styling

**Line 42:** `pad=20` in `set_title()` adds extra spacing between title and plot.

**Line 43:** `framealpha=0.9` in `legend()` controls legend box transparency (0=invisible, 1=opaque).

---

## Key Matplotlib Concepts Summary

### Two Styles of Using Matplotlib

**1. Pyplot style (simpler, good for quick plots):**
```python
plt.plot(x, y)
plt.xlabel('X')
plt.show()
```

**2. Object-oriented style (better for complex figures):**
```python
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_xlabel('X')
plt.show()
```

### Common Color Options
- Names: 'red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'black', 'white'
- Short codes: 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w'
- Hex codes: '#FF5733', '#3498DB'
- RGB tuples: (0.5, 0.3, 0.8) - values between 0 and 1

### Common Line Styles
- '-' : solid line
- '--' : dashed line
- ':' : dotted line
- '-.' : dash-dot line

### Common Markers
- 'o' : circle
- 's' : square
- '^' : triangle up
- 'v' : triangle down
- 'D' : diamond
- '*' : star
- '+' : plus
- 'x' : cross

### Saving Figures
- PNG: `plt.savefig('plot.png', dpi=300)` - raster, good for web
- PDF: `plt.savefig('plot.pdf')` - vector, good for publications
- SVG: `plt.savefig('plot.svg')` - vector, good for editing
- JPG: `plt.savefig('plot.jpg', quality=95)` - raster, compressed

### Figure Size and DPI
- Figure size: `plt.figure(figsize=(width, height))` in inches
- DPI (dots per inch): 
  - 72-96: screen display
  - 150-200: good quality prints
  - 300: publication quality

## When to Use What

**Line plots:** Continuous data, time series, mathematical functions
**Scatter plots:** Relationship between two variables, correlation
**Histograms:** Distribution of single variable, frequency analysis
**Bar charts:** Categorical data, comparisons
**Error bars:** Showing measurement uncertainty, confidence intervals
**Subplots:** Comparing multiple related plots, showing different aspects of data

## Common Gotchas

1. **Forgetting `plt.show()`** - Plot won't display
2. **Not calling `plt.figure()` before `plt.plot()`** - Can't control figure size
3. **Using data coordinates in `text()` without considering scale** - Text appears off-screen
4. **Saving after `show()`** - Figure gets cleared, save blank image
5. **Overlapping labels in subplots** - Use `plt.tight_layout()`

---

## Practice Exercises

1. Create a line plot with 3 different mathematical functions and use different colors, line styles, and markers for each.

2. Make a scatter plot where point size represents a third variable and colors represent a fourth variable using a colormap.

3. Create a 2x2 subplot grid showing: (a) histogram, (b) scatter plot, (c) bar chart, (d) line plot with error bars.

4. Plot data with error bars and add annotations pointing to the maximum and minimum values.

5. Create an overlapping histogram comparing two datasets with different colors and transparency.

## Additional Resources

- Official Documentation: https://matplotlib.org/stable/contents.html
- Gallery: https://matplotlib.org/stable/gallery/index.html
- Colormap Reference: https://matplotlib.org/stable/tutorials/colors/colormaps.html
