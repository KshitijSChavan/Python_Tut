# NumPy - Complete Guide

## Introduction

NumPy (Numerical Python) is the fundamental package for scientific computing in Python. It provides a powerful N-dimensional array object and tools for working with these arrays. NumPy arrays are much faster and more memory-efficient than Python lists for numerical operations.

**Key Capabilities:**
- Fast N-dimensional array operations
- Mathematical functions (trigonometry, statistics, linear algebra)
- Broadcasting for operations on arrays of different shapes
- Tools for reading/writing array data to disk
- Random number generation
- Fourier transforms and linear algebra operations

**Why NumPy over Python lists?**
- 10-100x faster for numerical operations
- Uses less memory
- Vectorized operations (no loops needed)
- Integration with C/C++ and Fortran code

**Installation:**
```bash
pip install numpy
```

---

## Example 1: Array Creation and Basic Operations

```python
import numpy as np

# Different ways to create arrays
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr3 = np.zeros((3, 4))
arr4 = np.ones((2, 3, 4))
arr5 = np.arange(0, 20, 2)
arr6 = np.linspace(0, 10, 50)
arr7 = np.random.rand(3, 3)
arr8 = np.eye(4)

# Display arrays and their properties
print("1D Array:", arr1)
print("Shape:", arr1.shape, "| Dimensions:", arr1.ndim, "| Size:", arr1.size, "| Type:", arr1.dtype)
print()

print("2D Array:\n", arr2)
print("Shape:", arr2.shape, "| Dimensions:", arr2.ndim, "| Size:", arr2.size)
print()

print("Zeros array (3x4):\n", arr3)
print()

print("Ones array (2x3x4) - 3D:")
print("Shape:", arr4.shape, "First slice:\n", arr4[0])
print()

print("Arange (0 to 20, step 2):", arr5)
print()

print("Linspace (0 to 10, 50 points):", arr6[:10], "...")
print()

print("Random array (3x3):\n", arr7)
print()

print("Identity matrix (4x4):\n", arr8)
print()

# Basic arithmetic operations (vectorized)
a = np.array([10, 20, 30, 40, 50])
b = np.array([1, 2, 3, 4, 5])

print("Array a:", a)
print("Array b:", b)
print("a + b:", a + b)
print("a - b:", a - b)
print("a * b:", a * b)
print("a / b:", a / b)
print("a ** 2:", a ** 2)
print("a > 25:", a > 25)
print()

# Universal functions
x = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])
print("x (in radians):", x)
print("sin(x):", np.sin(x))
print("cos(x):", np.cos(x))
print("tan(x):", np.tan(x))
print("exp(x):", np.exp(x))
print("log(x+1):", np.log(x + 1))
print("sqrt(x):", np.sqrt(x))
```

### Code Explanation:

**Line 4:** `np.array([1, 2, 3, 4, 5])` creates a 1D array from a Python list. NumPy automatically infers the data type (integers in this case).

**Line 5:** `np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])` creates a 2D array (matrix) from nested lists. Each inner list becomes a row.

**Line 6:** `np.zeros((3, 4))` creates a 3x4 array filled with zeros. The tuple `(3, 4)` specifies shape (rows, columns). Default data type is float64.

**Line 7:** `np.ones((2, 3, 4))` creates a 3D array with shape 2x3x4 filled with ones. First dimension is "depth", second is rows, third is columns.

**Line 8:** `np.arange(0, 20, 2)` creates an array with values from 0 to 20 (exclusive) with step size 2. Similar to Python's `range()` but returns a NumPy array: [0, 2, 4, 6, ..., 18].

**Line 9:** `np.linspace(0, 10, 50)` creates 50 evenly spaced numbers between 0 and 10 (inclusive). Unlike `arange`, you specify the number of points, not the step size. Useful for plotting smooth curves.

**Line 10:** `np.random.rand(3, 3)` creates a 3x3 array with random numbers uniformly distributed between 0 and 1.

**Line 11:** `np.eye(4)` creates a 4x4 identity matrix (1s on diagonal, 0s elsewhere). Useful in linear algebra.

**Line 14:** `arr1.shape` returns a tuple with dimensions: `(5,)` for 1D array with 5 elements.

**Line 14:** `arr1.ndim` returns number of dimensions (1 for 1D, 2 for 2D, etc.).

**Line 14:** `arr1.size` returns total number of elements (product of all dimensions).

**Line 14:** `arr1.dtype` returns data type of array elements (int64, float64, etc.).

**Line 18:** For 2D array with shape `(3, 3)`, first number is rows, second is columns.

**Line 38-43:** Arithmetic operations work element-wise (vectorized):
- `a + b` adds corresponding elements: [10+1, 20+2, 30+3, 40+4, 50+5]
- `a * b` multiplies element-wise (NOT matrix multiplication)
- `a ** 2` squares each element
- No loops needed - operations are fast and applied to all elements at once

**Line 44:** `a > 25` returns a boolean array: [False, False, True, True, True]. Each element is compared individually.

**Line 48-54:** Universal functions (ufuncs) apply mathematical operations element-wise:
- `np.sin(x)`, `np.cos(x)`, `np.tan(x)` - trigonometric functions (input in radians)
- `np.exp(x)` - exponential function (e^x)
- `np.log(x)` - natural logarithm
- `np.sqrt(x)` - square root
All operate on entire arrays without loops.

---

## Example 2: Array Indexing, Slicing, and Boolean Masking

```python
import numpy as np

# Create sample arrays
arr = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
matrix = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12],
                   [13, 14, 15, 16]])

print("Original 1D array:", arr)
print()

# Basic indexing (0-based)
print("First element:", arr[0])
print("Last element:", arr[-1])
print("Third element:", arr[2])
print()

# Slicing
print("First 5 elements:", arr[:5])
print("Last 3 elements:", arr[-3:])
print("Elements from index 2 to 7:", arr[2:8])
print("Every other element:", arr[::2])
print("Reverse array:", arr[::-1])
print()

# 2D array indexing
print("Original matrix:\n", matrix)
print()
print("Element at row 1, col 2:", matrix[1, 2])
print("First row:", matrix[0, :])
print("Second column:", matrix[:, 1])
print("Top-left 2x2 block:\n", matrix[:2, :2])
print("Bottom-right 2x2 block:\n", matrix[2:, 2:])
print("Middle 2x2 block:\n", matrix[1:3, 1:3])
print()

# Boolean masking (filtering)
print("Boolean mask (arr > 50):", arr > 50)
print("Elements greater than 50:", arr[arr > 50])
print("Elements between 30 and 70:", arr[(arr >= 30) & (arr <= 70)])
print("Elements less than 30 OR greater than 70:", arr[(arr < 30) | (arr > 70)])
print()

# Fancy indexing (using arrays of indices)
indices = np.array([0, 2, 5, 8])
print("Elements at indices [0, 2, 5, 8]:", arr[indices])
print()

# Modifying arrays
arr_copy = arr.copy()
arr_copy[arr_copy > 50] = 50  # Cap values at 50
print("Array with values capped at 50:", arr_copy)
print()

# 2D boolean masking
print("Matrix elements greater than 8:\n", matrix[matrix > 8])
matrix_copy = matrix.copy()
matrix_copy[matrix_copy % 2 == 0] = 0  # Set even numbers to 0
print("Matrix with even numbers set to 0:\n", matrix_copy)
print()

# Where function (conditional selection)
arr2 = np.array([15, 25, 35, 45, 55, 65])
result = np.where(arr2 > 40, arr2, 0)  # Keep if >40, else set to 0
print("Original:", arr2)
print("Where arr2 > 40:", result)
print()

# Replace values
arr3 = np.array([1, 2, 3, 4, 5])
arr3_modified = np.where(arr3 > 3, arr3 * 10, arr3)
print("Original:", arr3)
print("Multiply by 10 if >3:", arr3_modified)
```

### Code Explanation:

**Line 4:** Create 1D array for indexing demonstrations.

**Line 5-8:** Create 2D array (4x4 matrix) using nested lists.

**Line 13:** `arr[0]` accesses first element. NumPy uses 0-based indexing like Python lists.

**Line 14:** `arr[-1]` accesses last element. Negative indices count from the end: -1 is last, -2 is second-to-last, etc.

**Line 19:** `arr[:5]` slices from start to index 5 (exclusive). Returns [10, 20, 30, 40, 50].

**Line 20:** `arr[-3:]` slices last 3 elements. Returns [80, 90, 100].

**Line 21:** `arr[2:8]` slices from index 2 to 8 (exclusive). Start inclusive, stop exclusive.

**Line 22:** `arr[::2]` uses step size of 2. Format is `[start:stop:step]`. Returns every other element: [10, 30, 50, 70, 90].

**Line 23:** `arr[::-1]` reverses array. Negative step goes backwards.

**Line 29:** `matrix[1, 2]` accesses element at row 1, column 2 (value is 7). Use comma to separate dimensions.

**Line 30:** `matrix[0, :]` selects entire first row. `:` means "all elements in this dimension". Returns [1, 2, 3, 4].

**Line 31:** `matrix[:, 1]` selects entire second column (all rows, column 1). Returns [2, 6, 10, 14].

**Line 32:** `matrix[:2, :2]` slices first 2 rows and first 2 columns. Creates a 2x2 sub-array.

**Line 33:** `matrix[2:, 2:]` slices from row 2 to end and column 2 to end. Bottom-right block.

**Line 34:** `matrix[1:3, 1:3]` extracts middle 2x2 block (rows 1-2, columns 1-2).

**Line 38:** `arr > 50` creates boolean array: [False, False, False, False, False, True, True, True, True, True].

**Line 39:** `arr[arr > 50]` uses boolean array as mask. Returns only elements where mask is True: [60, 70, 80, 90, 100].

**Line 40:** `(arr >= 30) & (arr <= 70)` combines conditions with `&` (AND). Parentheses required around each condition. Cannot use `and`.

**Line 41:** `(arr < 30) | (arr > 70)` combines with `|` (OR operator). Cannot use `or`.

**Line 45:** Fancy indexing: pass array of indices to select multiple specific elements at once.

**Line 49:** `arr_copy = arr.copy()` creates independent copy. Without `.copy()`, would just create reference to same array.

**Line 50:** `arr_copy[arr_copy > 50] = 50` modifies elements in-place. Boolean indexing on left side of assignment sets matching elements to 50.

**Line 55:** `matrix[matrix > 8]` returns 1D array of elements that match condition, not preserving original 2D structure.

**Line 57:** `matrix_copy[matrix_copy % 2 == 0] = 0` sets all even numbers to 0. `%` is modulo operator.

**Line 61:** `np.where(condition, x, y)` returns x where condition is True, y where False. Like a vectorized if-else statement.
- First argument: boolean condition
- Second argument: value if True
- Third argument: value if False

**Line 68:** `np.where()` can also transform values: multiply by 10 if condition true, keep original otherwise.

---

## Example 3: Array Reshaping, Stacking, and Splitting

```python
import numpy as np

# Create sample arrays
arr1d = np.arange(12)
arr2d = np.arange(12).reshape(3, 4)
arr3d = np.arange(24).reshape(2, 3, 4)

print("Original 1D array:", arr1d)
print("Shape:", arr1d.shape)
print()

# Reshaping
reshaped_3x4 = arr1d.reshape(3, 4)
print("Reshaped to 3x4:\n", reshaped_3x4)
print()

reshaped_2x6 = arr1d.reshape(2, 6)
print("Reshaped to 2x6:\n", reshaped_2x6)
print()

reshaped_4x3 = arr1d.reshape(4, 3)
print("Reshaped to 4x3:\n", reshaped_4x3)
print()

# Reshaping with -1 (automatic dimension calculation)
auto_reshape = arr1d.reshape(3, -1)  # 3 rows, automatically calculate columns
print("Reshape with -1 (3, -1):\n", auto_reshape)
print()

# Flattening (2D to 1D)
print("2D array:\n", arr2d)
flat = arr2d.flatten()
print("Flattened:", flat)
ravel = arr2d.ravel()
print("Raveled:", ravel)
print()

# Transpose
print("Original 2D:\n", arr2d)
print("Transposed:\n", arr2d.T)
print()

# Stacking arrays vertically (row-wise)
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
print("Array a:\n", a)
print("Array b:\n", b)
print()

vstacked = np.vstack((a, b))
print("Vertical stack (vstack):\n", vstacked)
print()

# Stacking arrays horizontally (column-wise)
hstacked = np.hstack((a, b))
print("Horizontal stack (hstack):\n", hstacked)
print()

# Stacking with concatenate
concat_axis0 = np.concatenate((a, b), axis=0)  # Along rows (vertical)
concat_axis1 = np.concatenate((a, b), axis=1)  # Along columns (horizontal)
print("Concatenate axis=0:\n", concat_axis0)
print("Concatenate axis=1:\n", concat_axis1)
print()

# Splitting arrays
arr_to_split = np.arange(16).reshape(4, 4)
print("Array to split:\n", arr_to_split)
print()

# Horizontal split (along columns)
left, right = np.hsplit(arr_to_split, 2)
print("Horizontal split (2 parts):")
print("Left:\n", left)
print("Right:\n", right)
print()

# Vertical split (along rows)
top, bottom = np.vsplit(arr_to_split, 2)
print("Vertical split (2 parts):")
print("Top:\n", top)
print("Bottom:\n", bottom)
print()

# Split at specific indices
arr_to_split2 = np.arange(12)
part1, part2, part3 = np.split(arr_to_split2, [4, 8])
print("Array:", arr_to_split2)
print("Split at indices [4, 8]:")
print("Part 1:", part1)
print("Part 2:", part2)
print("Part 3:", part3)
print()

# Adding new axis
arr_1d = np.array([1, 2, 3, 4, 5])
print("1D array shape:", arr_1d.shape, "Array:", arr_1d)
arr_row = arr_1d[np.newaxis, :]
print("As row (1, 5):", arr_row.shape, "Array:", arr_row)
arr_col = arr_1d[:, np.newaxis]
print("As column (5, 1) shape:", arr_col.shape, "\nArray:\n", arr_col)
```

### Code Explanation:

**Line 4:** `np.arange(12)` creates [0, 1, 2, ..., 11].

**Line 5:** `.reshape(3, 4)` converts 1D array to 2D with 3 rows and 4 columns. Total elements must match (12 = 3×4).

**Line 6:** `.reshape(2, 3, 4)` creates 3D array. Total elements: 2×3×4 = 24.

**Line 13:** `reshape()` returns a new view (doesn't copy data). Changes shape but not data. New shape must have same number of total elements.

**Line 25:** `.reshape(3, -1)` uses -1 to automatically calculate dimension. With 3 rows and 12 total elements, NumPy calculates 4 columns (12/3 = 4). Only one dimension can be -1.

**Line 30:** `.flatten()` converts any shape to 1D array. Always returns a copy (new array in memory).

**Line 32:** `.ravel()` also converts to 1D but returns a view when possible (doesn't copy). Faster but changes to ravel affect original array.

**Line 37:** `.T` transposes the array (swaps rows and columns). For 2D array, row becomes column and vice versa. For 3D+, swaps first and last axes.

**Line 47:** `np.vstack((a, b))` stacks arrays vertically (adds rows). Arrays must have same number of columns. Result has 4 rows (2+2) and 2 columns.

**Line 52:** `np.hstack((a, b))` stacks arrays horizontally (adds columns). Arrays must have same number of rows. Result has 2 rows and 4 columns (2+2).

**Line 56:** `np.concatenate((a, b), axis=0)` joins arrays along specified axis:
- `axis=0` - concatenate along rows (vertical, same as vstack)
- `axis=1` - concatenate along columns (horizontal, same as hstack)
More general than vstack/hstack, works with any number of dimensions.

**Line 65:** `np.hsplit(arr, 2)` splits array horizontally into 2 equal parts. Number of columns must be divisible by split number.

**Line 72:** `np.vsplit(arr, 2)` splits array vertically into 2 equal parts. Number of rows must be divisible by split number.

**Line 79:** `np.split(arr, [4, 8])` splits at specified indices:
- Indices [4, 8] create splits: [:4], [4:8], [8:]
- Returns 3 arrays: elements 0-3, elements 4-7, elements 8-11

**Line 89:** `arr[np.newaxis, :]` adds new axis at beginning. Converts shape (5,) to (1, 5) - makes it a row vector.

**Line 91:** `arr[:, np.newaxis]` adds new axis at end. Converts shape (5,) to (5, 1) - makes it a column vector. Useful for broadcasting operations.

---

## Example 4: Statistical Operations and Aggregations

```python
import numpy as np

# Create sample data
data = np.array([15, 23, 18, 42, 31, 29, 17, 38, 25, 20])
data_2d = np.array([[10, 20, 30, 40],
                    [15, 25, 35, 45],
                    [12, 22, 32, 42]])

print("1D Data:", data)
print("2D Data:\n", data_2d)
print()

# Basic statistics
print("=== Basic Statistics (1D) ===")
print("Mean:", np.mean(data))
print("Median:", np.median(data))
print("Standard deviation:", np.std(data))
print("Variance:", np.var(data))
print("Min:", np.min(data))
print("Max:", np.max(data))
print("Sum:", np.sum(data))
print("Product:", np.prod(data))
print()

# Percentiles
print("=== Percentiles ===")
print("25th percentile (Q1):", np.percentile(data, 25))
print("50th percentile (Median):", np.percentile(data, 50))
print("75th percentile (Q3):", np.percentile(data, 75))
print("10th and 90th percentiles:", np.percentile(data, [10, 90]))
print()

# Cumulative operations
print("=== Cumulative Operations ===")
print("Original:", data)
print("Cumulative sum:", np.cumsum(data))
print("Cumulative product:", np.cumprod(data[:5]))  # First 5 to avoid overflow
print()

# Axis-wise operations on 2D arrays
print("=== Axis-wise Operations (2D) ===")
print("Array:\n", data_2d)
print()
print("Mean of entire array:", np.mean(data_2d))
print("Mean along axis=0 (down columns):", np.mean(data_2d, axis=0))
print("Mean along axis=1 (across rows):", np.mean(data_2d, axis=1))
print()
print("Sum along axis=0:", np.sum(data_2d, axis=0))
print("Sum along axis=1:", np.sum(data_2d, axis=1))
print()
print("Max along axis=0:", np.max(data_2d, axis=0))
print("Max along axis=1:", np.max(data_2d, axis=1))
print()

# Argmin and Argmax (indices of min/max)
print("=== Finding Indices ===")
print("Index of minimum:", np.argmin(data))
print("Index of maximum:", np.argmax(data))
print("Value at min index:", data[np.argmin(data)])
print("Value at max index:", data[np.argmax(data)])
print()

# 2D argmin/argmax
print("2D Array:\n", data_2d)
print("Index of min (flattened):", np.argmin(data_2d))
print("Index of max (flattened):", np.argmax(data_2d))
print("Index of min along axis=0:", np.argmin(data_2d, axis=0))
print("Index of min along axis=1:", np.argmin(data_2d, axis=1))
print()

# Sorting
print("=== Sorting ===")
unsorted = np.array([23, 12, 45, 18, 34, 29, 8, 37])
print("Original:", unsorted)
print("Sorted:", np.sort(unsorted))
print("Indices that would sort array:", np.argsort(unsorted))
sorted_indices = np.argsort(unsorted)
print("Array sorted by indices:", unsorted[sorted_indices])
print()

# 2D sorting
arr_2d = np.array([[3, 1, 4], [9, 2, 6], [5, 8, 7]])
print("2D Array:\n", arr_2d)
print("Sorted along axis=0:\n", np.sort(arr_2d, axis=0))
print("Sorted along axis=1:\n", np.sort(arr_2d, axis=1))
print()

# Unique values
data_with_duplicates = np.array([1, 2, 2, 3, 3, 3, 4, 4, 5])
print("Array with duplicates:", data_with_duplicates)
print("Unique values:", np.unique(data_with_duplicates))
unique, counts = np.unique(data_with_duplicates, return_counts=True)
print("Unique values with counts:")
for val, count in zip(unique, counts):
    print(f"  {val}: appears {count} times")
print()

# Clipping values
data_to_clip = np.array([5, 15, 25, 35, 45, 55, 65, 75])
print("Original:", data_to_clip)
print("Clipped [20, 60]:", np.clip(data_to_clip, 20, 60))
print()

# Rounding
floats = np.array([1.23456, 2.34567, 3.45678, 4.56789])
print("Original floats:", floats)
print("Rounded to 2 decimals:", np.round(floats, 2))
print("Floor:", np.floor(floats))
print("Ceiling:", np.ceil(floats))
```

### Code Explanation:

**Line 4-7:** Create sample 1D and 2D arrays for statistical demonstrations.

**Line 14:** `np.mean(data)` calculates arithmetic mean (average). Sums all elements and divides by count.

**Line 15:** `np.median(data)` finds middle value when sorted. For even-length arrays, averages two middle values.

**Line 16:** `np.std(data)` calculates standard deviation (measure of spread). By default computes population std (divides by N). Use `ddof=1` for sample std (divides by N-1).

**Line 17:** `np.var(data)` calculates variance (squared standard deviation).

**Line 18-19:** `np.min()` and `np.max()` find minimum and maximum values.

**Line 20:** `np.sum(data)` adds all elements. Much faster than Python's `sum()` for large arrays.

**Line 21:** `np.prod(data)` multiplies all elements together. Can overflow for large arrays.

**Line 25-27:** `np.percentile(data, q)` finds value below which q% of data falls:
- 25th percentile (Q1) - 25% of data is below this
- 50th percentile - median
- 75th percentile (Q3) - 75% of data is below this

**Line 28:** Can pass list of percentiles to get multiple values at once.

**Line 33:** `np.cumsum(data)` returns cumulative sum: [15, 15+23, 15+23+18, ...]. Each element is sum of all previous elements plus itself.

**Line 34:** `np.cumprod(data)` returns cumulative product. Can overflow quickly, so limited to first 5 elements here.

**Line 40:** `np.mean(data_2d)` without axis computes mean of all elements in entire array.

**Line 41:** `np.mean(data_2d, axis=0)` computes mean down each column (along rows). Returns array with one value per column.
- `axis=0` means "collapse the 0th dimension (rows)"
- Result shape: (4,) - one mean per column

**Line 42:** `np.mean(data_2d, axis=1)` computes mean across each row. Returns array with one value per row.
- `axis=1` means "collapse the 1st dimension (columns)"
- Result shape: (3,) - one mean per row

**Line 54:** `np.argmin(data)` returns index of minimum value (not the value itself). For 1D array, returns single integer.

**Line 55:** `np.argmax(data)` returns index of maximum value.

**Line 62:** For 2D array without axis, `argmin` treats array as flattened and returns single index.

**Line 64:** `argmin(data_2d, axis=0)` returns index of minimum in each column. Result has one index per column.

**Line 65:** `argmin(data_2d, axis=1)` returns index of minimum in each row. Result has one index per row.

**Line 71:** `np.sort(unsorted)` returns sorted copy (doesn't modify original). Default is ascending order. Use `[::-1]` for descending.

**Line 72:** `np.argsort(unsorted)` returns indices that would sort the array. Useful when you need to sort multiple arrays the same way.

**Line 79:** Sorting 2D array along `axis=0` sorts each column independently.

**Line 80:** Sorting along `axis=1` sorts each row independently.

**Line 85:** `np.unique(data)` returns sorted unique values, removing duplicates.

**Line 86:** `return_counts=True` also returns count of each unique value.

**Line 93:** `np.clip(data, min, max)` limits values to range [min, max]. Values below min become min, values above max become max.

**Line 98:** `np.round(floats, decimals)` rounds to specified decimal places.

**Line 99:** `np.floor(floats)` rounds down to nearest integer. 2.7 becomes 2.

**Line 100:** `np.ceil(floats)` rounds up to nearest integer. 2.3 becomes 3.

---

## Example 5: Broadcasting and Advanced Array Operations

```python
import numpy as np

# Broadcasting: operations between arrays of different shapes
print("=== Broadcasting Examples ===")

# Example 1: Scalar and array
arr = np.array([1, 2, 3, 4, 5])
print("Array:", arr)
print("Array + 10:", arr + 10)  # Scalar broadcast to all elements
print("Array * 2:", arr * 2)
print()

# Example 2: 1D array and 2D array
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
row_vec = np.array([10, 20, 30])

print("Matrix:\n", matrix)
print("Row vector:", row_vec)
print("Matrix + row_vec:\n", matrix + row_vec)  # row_vec broadcast to each row
print()

# Example 3: Column vector and 2D array
col_vec = np.array([[100], [200], [300]])
print("Column vector:\n", col_vec)
print("Matrix + col_vec:\n", matrix + col_vec)  # col_vec broadcast to each column
print()

# Example 4: Outer product using broadcasting
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30])
outer = a[:, np.newaxis] * b[np.newaxis, :]
print("Vector a:", a)
print("Vector b:", b)
print("Outer product:\n", outer)
print()

# Linear algebra operations
print("=== Linear Algebra ===")
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
v = np.array([1, 2])

print("Matrix A:\n", A)
print("Matrix B:\n", B)
print("Vector v:", v)
print()

# Matrix multiplication
print("A @ B (matrix multiplication):\n", A @ B)
print("A.dot(B) (same as @):\n", A.dot(B))
print()

# Matrix-vector multiplication
print("A @ v (matrix-vector):", A @ v)
print()

# Element-wise multiplication (NOT matrix multiplication)
print("A * B (element-wise):\n", A * B)
print()

# Determinant, inverse, eigenvalues
print("Determinant of A:", np.linalg.det(A))
print("Inverse of A:\n", np.linalg.inv(A))
print("A @ inv(A) (should be identity):\n", A @ np.linalg.inv(A))
eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)
print()

# Solving linear system Ax = b
b = np.array([5, 11])
x = np.linalg.solve(A, b)
print("Solving Ax = b where b =", b)
print("Solution x:", x)
print("Verification A @ x:", A @ x)
print()

# Random number generation
print("=== Random Numbers ===")
np.random.seed(42)  # For reproducibility

print("5 random floats [0, 1):", np.random.rand(5))
print("5 random integers [1, 100]:", np.random.randint(1, 101, 5))
print("5 random from normal distribution:", np.random.randn(5))
print("5 random from normal(50, 10):", np.random.normal(50, 10, 5))
print("3x3 random matrix:\n", np.random.rand(3, 3))
print()

# Random sampling
choices = np.array(['A', 'B', 'C', 'D'])
print("Random choice from", choices, ":", np.random.choice(choices))
print("5 random choices:", np.random.choice(choices, 5))
print("5 unique random choices:", np.random.choice(choices, 3, replace=False))
print()

# Shuffling
deck = np.arange(52)
print("Deck:", deck[:10], "...")
np.random.shuffle(deck)
print("Shuffled:", deck[:10], "...")
print()

# Set operations
print("=== Set Operations ===")
a = np.array([1, 2, 3, 4, 5])
b = np.array([4, 5, 6, 7, 8])
print("Array a:", a)
print("Array b:", b)
print("Intersection:", np.intersect1d(a, b))
print("Union:", np.union1d(a, b))
print("Set difference (a - b):", np.setdiff1d(a, b))
print("Set difference (b - a):", np.setdiff1d(b, a))
print()

# Conditional operations
print("=== Advanced Conditional Operations ===")
data = np.array([10, -5, 20, -3, 15, -8, 25, -2])
print("Data:", data)
print("Absolute values:", np.abs(data))
print("Sign (-1, 0, or 1):", np.sign(data))
print("Replace negatives with 0:", np.maximum(data, 0))
print("Replace values < 15 with 15:", np.maximum(data, 15))
```

### Code Explanation:

**Line 7-9:** Broadcasting automatically extends scalars to match array shape. `arr + 10` adds 10 to every element without explicit loop.

**Line 19:** Broadcasting 1D array to 2D: `row_vec` shape (3,) is broadcast to match `matrix` shape (3, 3). The row vector is "copied" to each row of the matrix mentally (no actual copying in memory).

**Line 25:** `col_vec` shape (3, 1) broadcasts along columns. Each element of col_vec is added to corresponding row of matrix.

**Line 32:** Outer product using broadcasting: reshape `a` to column (4, 1) and `b` to row (1, 3). Broadcasting creates (4, 3) result where result[i, j] = a[i] * b[j].

**Broadcasting Rules:**
1. If arrays have different dimensions, prepend 1s to smaller shape
2. Arrays compatible if dimensions are equal or one is 1
3. After broadcasting, each array behaves as if it had shape of the larger array

**Line 50:** `@` operator performs matrix multiplication (Python 3.5+). Result[i,j] = sum(A[i,:] * B[:,j]).

**Line 51:** `.dot()` method also does matrix multiplication. Older alternative to `@`.

**Line 55:** Matrix-vector multiplication: (2, 2) @ (2,) gives (2,). Each row of A multiplies vector v.

**Line 59:** `*` operator is element-wise multiplication, NOT matrix multiplication. A * B multiplies corresponding elements.

**Line 63:** `np.linalg.det(A)` calculates determinant. For 2x2 matrix [[a,b],[c,d]], det = ad - bc.

**Line 64:** `np.linalg.inv(A)` computes matrix inverse. A @ inv(A) = identity matrix.

**Line 66:** `np.linalg.eig(A)` returns eigenvalues and eigenvectors. Eigenvalue λ and eigenvector v satisfy: A @ v = λ * v.

**Line 72:** `np.linalg.solve(A, b)` solves linear system Ax = b for x. More numerically stable than computing x = inv(A) @ b.

**Line 79:** `np.random.seed(42)` sets random seed for reproducibility. Same seed gives same sequence of "random" numbers.

**Line 81:** `np.random.rand(5)` generates 5 random floats uniformly distributed in [0, 1).

**Line 82:** `np.random.randint(low, high, size)` generates random integers in [low, high). Note: high is exclusive.

**Line 83:** `np.random.randn(5)` generates from standard normal distribution (mean=0, std=1).

**Line 84:** `np.random.normal(mean, std, size)` generates from normal distribution with specified mean and standard deviation.

**Line 90:** `np.random.choice(array, size)` randomly samples from array. 

**Line 91:** `replace=False` ensures no duplicates (sampling without replacement). Size must be ≤ array length.

**Line 96:** `np.random.shuffle(array)` shuffles array in-place (modifies original).

**Line 103:** `np.intersect1d(a, b)` returns sorted common elements (set intersection).

**Line 104:** `np.union1d(a, b)` returns sorted unique elements from both arrays (set union).

**Line 105-106:** `np.setdiff1d(a, b)` returns elements in a but not in b (set difference).

**Line 112:** `np.abs(data)` returns absolute values (|-5| = 5).

**Line 113:** `np.sign(data)` returns -1 for negative, 0 for zero, +1 for positive.

**Line 114:** `np.maximum(data, 0)` element-wise maximum between data and 0. Clips negative values to 0.

**Line 115:** `np.maximum(data, 15)` sets minimum value to 15 (values below 15 become 15).

---

## Key NumPy Concepts Summary

### Array Creation Methods
```python
np.array([1,2,3])           # From list
np.zeros((3,4))             # Array of zeros
np.ones((2,3))              # Array of ones
np.arange(0, 10, 2)         # Range with step
np.linspace(0, 10, 50)      # Evenly spaced
np.eye(4)                   # Identity matrix
np.random.rand(3,3)         # Random [0,1)
```

### Data Types (dtype)
- `int8`, `int16`, `int32`, `int64` - integers
- `float16`, `float32`, `float64` - floating point
- `bool` - boolean
- `complex64`, `complex128` - complex numbers
- Specify: `np.array([1,2,3], dtype=np.float32)`

### Axes in NumPy
- `axis=0` - operates down rows (column-wise)
- `axis=1` - operates across columns (row-wise)
- `axis=None` - operates on flattened array

### Memory: Views vs Copies
- **View**: shares data with original (changes affect both)
  - `arr[:]`, `arr.reshape()`, `arr.T`, `arr.ravel()`
- **Copy**: independent data (changes don't affect original)
  - `arr.copy()`, `arr.flatten()`

### Broadcasting Rules
1. Start with trailing dimensions
2. Dimensions compatible if equal or one is 1
3. Missing dimensions treated as 1

### Common Operations Cheat Sheet
```python
# Statistics
arr.mean(), arr.std(), arr.var()
arr.min(), arr.max(), arr.sum()
np.median(arr), np.percentile(arr, 75)

# Shape
arr.shape, arr.ndim, arr.size
arr.reshape(3,4), arr.flatten(), arr.T

# Indexing
arr[0], arr[-1], arr[1:5]
arr[arr > 0], arr[[0,2,4]]

# Math
np.sin(arr), np.cos(arr), np.exp(arr)
np.sqrt(arr), np.log(arr), np.abs(arr)

# Aggregation
np.sum(arr, axis=0), np.mean(arr, axis=1)
np.cumsum(arr), np.cumprod(arr)
```

## When to Use NumPy

**Use NumPy when:**
- Working with numerical data
- Need fast array operations
- Doing linear algebra, statistics
- Processing large datasets
- Need multidimensional arrays

**NumPy faster than lists for:**
- Mathematical operations
- Element-wise operations
- Aggregations (sum, mean, etc.)
- Large datasets (1000+ elements)

## Common Gotchas

1. **Views vs Copies**: `arr2 = arr1` creates view, not copy. Changes to arr2 affect arr1. Use `.copy()` for independent copy.

2. **Integer division**: `np.array([1,2,3]) / 2` gives floats, but `np.array([1,2,3]) // 2` gives integers.

3. **In-place operations**: `arr += 1` modifies arr, but `arr = arr + 1` creates new array.

4. **Axis confusion**: `axis=0` operates along first dimension (often confusing for 2D arrays).

5. **Broadcasting errors**: "operands could not be broadcast together" means shapes incompatible.

---

## Practice Exercises

1. Create a 5x5 matrix of random integers 1-100, find all values > 50, and replace them with 50.

2. Create two 1D arrays and compute their outer product using broadcasting.

3. Load a 2D array and compute mean and std for each row, then normalize each row (subtract mean, divide by std).

4. Create a 4x4 matrix, extract the 2x2 center block, and compute its determinant.

5. Generate 1000 random numbers from normal distribution, create histogram with 30 bins, find what percentage fall within 1 standard deviation of mean.

## Additional Resources

- Official Documentation: https://numpy.org/doc/stable/
- NumPy Quickstart: https://numpy.org/doc/stable/user/quickstart.html
- Broadcasting: https://numpy.org/doc/stable/user/basics.broadcasting.html
- Linear Algebra: https://numpy.org/doc/stable/reference/routines.linalg.html
