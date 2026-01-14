# NumPy Basics

Python lists work fine for small datasets, but when you're processing thousands of sources or large images, they're too slow. **NumPy** (Numerical Python) provides fast arrays optimized for scientific computing. It's the foundation of almost all scientific Python code.

Think of NumPy arrays like lists, but supercharged - they're 10-100x faster, use less memory, and have built-in functions for all the statistics we calculated manually. The speed difference becomes dramatic with large datasets: calculating the mean of a million numbers takes 100 milliseconds with a list, but only 1 millisecond with NumPy.

## Do You Have NumPy?

First, check if NumPy is already installed. Open a terminal and try:

```bash
python3 -c "import numpy; print(numpy.__version__)"
```

If you see a version number (like `1.24.3`), you're good to go. If you get `ModuleNotFoundError: No module named 'numpy'`, you need to install it:

```bash
pip3 install numpy --break-system-packages
# or
pip install numpy --break-system-packages
```

The `--break-system-packages` flag is often needed on newer systems. If you still have issues, see `00_common_errors_and_solutions.md`.

## Creating Arrays

The most common way is converting a list. You import NumPy (usually as `np` - that's the convention everyone uses), then create an array:

```python
import numpy as np

flux_list = [245.7, 189.3, 312.5, 198.7, 267.4]
flux_array = np.array(flux_list)
print(flux_array)  # [245.7 189.3 312.5 198.7 267.4]
```

Notice the output looks similar to a list but without commas. This is a NumPy array.

You can also create arrays directly without starting from a list. Need an array of zeros? Use `np.zeros(5)`. Need ones? `np.ones(3)`. Need a range? `np.arange(100, 1001, 100)` gives you 100, 200, 300... up to 1000. 

There's a subtle difference between `arange` and `linspace` worth knowing: `arange(0, 10, 2)` creates values with step size 2 (so you get 0, 2, 4, 6, 8), while `linspace(0, 10, 5)` creates exactly 5 evenly-spaced points between 0 and 10 (so you get 0, 2.5, 5, 7.5, 10). Use `arange` when you know the step size, `linspace` when you know how many points you need.

## Why NumPy Is Fast

The magic of NumPy is **vectorization** - you can do math on entire arrays without writing loops. Want to convert all flux values from mJy to Jy? Just divide the array by 1000:

```python
flux = np.array([245.7, 189.3, 312.5, 198.7, 267.4])
flux_Jy = flux / 1000
print(flux_Jy)  # [0.2457 0.1893 0.3125 0.1987 0.2674]
```

This operation happens in compiled C code under the hood, which is why it's so fast. Compare this to a loop where you'd divide each element one by one - NumPy does them all at once.

You can do element-wise operations between arrays too. Got flux measurements at two epochs? Subtract them to see variability:

```python
flux_epoch1 = np.array([245.7, 189.3, 312.5])
flux_epoch2 = np.array([250.0, 195.0, 305.0])
change = flux_epoch1 - flux_epoch2
print(change)  # [-4.3 -5.7  7.5] mJy
```

## Statistics Built-In

Remember all those functions we wrote for mean, median, standard deviation? NumPy has them all:

```python
flux = np.array([245.7, 189.3, 312.5, 198.7, 267.4])

mean = np.mean(flux)
median = np.median(flux)
std = np.std(flux, ddof=1)  # ddof=1 for sample std (N-1)
print(f"Mean: {mean:.2f}, Median: {median:.2f}, Std: {std:.2f}")
```

The `ddof=1` parameter is important - it uses N-1 in the denominator (sample standard deviation) instead of N (population). Without it, NumPy assumes you have the entire population. In astronomy, you're usually analyzing a sample, so use `ddof=1`.

You also get `np.min()`, `np.max()`, `np.sum()`, and `np.percentile()` - all the statistics we need.

## Boolean Indexing - The Killer Feature

Here's where NumPy really shines. You can filter arrays based on conditions without writing any loops. Say you want only the bright sources (flux > 200 mJy):

```python
flux = np.array([245.7, 89.3, 312.5, 45.2, 267.4])
bright = flux[flux > 200]
print(bright)  # [245.7 312.5 267.4]
```

What's happening? `flux > 200` creates a boolean array (True/False for each element), then that boolean array is used to select elements. This is called **masking** and it's incredibly powerful.

You can combine conditions using `&` (and), `|` (or), and `~` (not). Important: use these symbols, not the words `and`/`or`/`not` - those don't work with arrays. Also, you need parentheses around each condition:

```python
flux = np.array([245.7, 89.3, 312.5, 45.2, 267.4])
medium = flux[(flux > 100) & (flux < 300)]  # Between 100 and 300
print(medium)  # [245.7 267.4]
```

If you forget the parentheses, you'll get an error about operator precedence. If you use `and` instead of `&`, you'll get a `ValueError` saying "The truth value of an array is ambiguous."

## Working with 2D Data

NumPy handles tables naturally. No more nested lists - just create a 2D array:

```python
# 3 sources, 4 frequencies each
flux_table = np.array([
    [245.7, 189.3, 156.2, 134.5],
    [312.5, 278.9, 234.1, 198.3],
    [198.7, 167.4, 145.8, 125.2]
])
print(flux_table.shape)  # (3, 4) means 3 rows, 4 columns
```

Accessing data is elegant. Want all flux values for the first source? `flux_table[0]`. Want all flux values at the first frequency? `flux_table[:, 0]`. The colon means "all" - so `[:,0]` means "all rows, column 0". This is much cleaner than nested list comprehensions.

## Mathematical Constants and Basic Math

NumPy provides precise values for π and e:

```python
import numpy as np
area = np.pi * (30 ** 2)  # Circle area with 30 arcsec radius
```

Use `np.pi` instead of manually typing 3.14159 - it's more precise and everyone will know what you mean.

## Things Worth Noting

**Arrays enforce one type.** If you create an array from `[1, 2.5, 3]`, all elements become floats. NumPy can't mix integers and floats like lists can. This is actually a feature - it's how NumPy achieves its speed. If you try to create an array with numbers and strings, everything becomes a string, which probably isn't what you want.

**Parentheses matter with boolean operations.** Write `flux[(flux > 100) & (flux < 300)]`, not `flux[flux > 100 & flux < 300]`. The second version gives a cryptic error about operator precedence.

**Views vs copies can surprise you.** When you slice an array like `view = arr[:]`, you're creating a view, not a copy. Modify the view and the original changes too:

```python
arr = np.array([1, 2, 3])
view = arr[:]
view[0] = 999
print(arr)  # [999 2 3] - original changed!
```

If you need an independent copy, use `arr.copy()`.

**Use `&` not `and`.** For combining conditions on arrays, you must use `&` (and), `|` (or), `~` (not). The regular Python keywords `and`/`or`/`not` give a `ValueError` about ambiguous truth values.

## Try This

1. Create an array of frequencies from 100 to 1000 MHz in steps of 50
2. Convert an array of fluxes in mJy to Jy
3. Filter an array to keep only values between 100 and 300
4. Calculate mean and std of: `[245.7, 189.3, 312.5, 198.7, 267.4]` using NumPy
5. Create a 3x3 array of ones, then set the diagonal to zeros

## How This Is Typically Used in Astronomy

Processing large catalogs (millions of sources), manipulating images (2D arrays), fast calculations on spectra, filtering data by multiple criteria, and statistical analysis of large datasets.

Almost all astronomy Python packages (astropy, scipy, matplotlib) expect NumPy arrays.

## Related Lessons

**Previous**: [13_statistics_as_functions.md](13_statistics_as_functions.md) - Manual statistics

**Next**: [15_numpy_data_loading.md](15_numpy_data_loading.md) - Loading data with NumPy

**Replaces**: [07_nested_lists.md](07_nested_lists.md) - NumPy arrays are better for numerical 2D data
