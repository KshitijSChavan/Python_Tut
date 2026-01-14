# Nested Lists

Sometimes you need to store 2D data - like a grid of flux values, a table of sources with multiple properties, or an image represented as pixels. Python handles this with **nested lists** - lists inside lists.

## Creating 2D Lists

A nested list is just a list where each item is itself a list:

```python
# Flux measurements at different frequencies for 3 sources
# Each row is one source, each column is a frequency
flux_table = [
    [245.7, 189.3, 156.2],  # Source 1
    [312.5, 278.9, 234.1],  # Source 2
    [198.7, 167.4, 145.8]   # Source 3
]

print(flux_table)
# Output: [[245.7, 189.3, 156.2], [312.5, 278.9, 234.1], [198.7, 167.4, 145.8]]
```

Think of it as a table with rows and columns, like a spreadsheet.

## Accessing Elements

To get a specific value, you need to tell Python which row and which column. Use two sets of brackets - first for the row, then for the column:

```python
flux_table = [
    [245.7, 189.3, 156.2],
    [312.5, 278.9, 234.1],
    [198.7, 167.4, 145.8]
]

# Get the entire first row (all data for first source)
first_source = flux_table[0]
print(first_source)  # Output: [245.7, 189.3, 156.2]

# Get a specific element: second source, third frequency
value = flux_table[1][2]
print(value)  # Output: 234.1
```

Think of `flux_table[1][2]` as: "Go to row 1 (second source), then go to column 2 (third frequency)". Remember Python counts from 0.

You can modify values the same way:

```python
flux_table[0][1] = 195.0  # Update source 1, frequency 2
print(flux_table[0])  # Output: [245.7, 195.0, 156.2]
```

If you try to access an index that doesn't exist, you'll get an `IndexError`:

```python
# flux_table[5]  # IndexError! Only 3 rows (indices 0, 1, 2)
# flux_table[0][5]  # IndexError! Only 3 columns (indices 0, 1, 2)
```

## Getting Rows and Columns

Here's an asymmetry that takes getting used to: getting a row is easy, but getting a column takes more work.

Getting a row is straightforward - it's just one list:

```python
# Get all flux values for source 2
source_2_fluxes = flux_table[1]
print(source_2_fluxes)  # Output: [312.5, 278.9, 234.1]
```

But what if you want all the flux values at one frequency across all sources? That's a column, and you need to extract it:

```python
# Get all flux values at the first frequency (column 0)
first_freq_fluxes = [row[0] for row in flux_table]
print(first_freq_fluxes)  # Output: [245.7, 312.5, 198.7]
```

This says: "for each row, take the first element (column 0)". You can also write it as a traditional loop:

```python
first_freq_fluxes = []
for row in flux_table:
    first_freq_fluxes.append(row[0])
```

This asymmetry (rows easy, columns harder) is one reason NumPy arrays are better for numerical work.

## Building Nested Lists

Sometimes you start with an empty table and fill it as you go:

```python
# Start with empty table
observations = []

# Add rows one at a time
observations.append([245.7, 189.3, 156.2])
observations.append([312.5, 278.9, 234.1])
observations.append([198.7, 167.4, 145.8])

print(observations)
```

Other times you want to create a table of a specific size, filled with zeros or some default value:

```python
# Create 3x4 table filled with zeros
rows = 3
cols = 4
table = [[0 for j in range(cols)] for i in range(rows)]
print(table)
# Output: [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
```

Read this as: "make a list of rows, where each row is a list of cols zeros". The inner `[0 for j in range(cols)]` creates one row, and the outer list comprehension repeats this for each row.

Now here's a subtle trap that catches many people. This looks similar but behaves very differently:

```python
# WRONG way - creates references to same list!
table = [[0] * 3] * 3
table[0][0] = 999
print(table)  # Output: [[999, 0, 0], [999, 0, 0], [999, 0, 0]]
# Oops! All rows changed, not just the first one.
```

Why? Because `[[0] * 3] * 3` creates one list `[0, 0, 0]` and then creates three references to that same list. Changing one changes all of them. You'll get a surprise when you modify what you think is one row.

The right way creates separate lists for each row:

```python
# RIGHT way - creates separate lists
table = [[0] * 3 for i in range(3)]
table[0][0] = 999
print(table)  # Output: [[999, 0, 0], [0, 0, 0], [0, 0, 0]]
# Only first row changed, as expected
```

This is one of the most common bugs with nested lists. If you ever modify one element and see multiple elements change, you've hit this trap.

## Looping Through 2D Lists

There are several ways to loop through a nested list, depending on what you need:

If you just want to process each row:

```python
for row in flux_table:
    print(row)
```

If you need every individual element:

```python
for row in flux_table:
    for value in row:
        print(value)
```

Sometimes you need to know which row and column you're in. Use `range()` with indices:

```python
for i in range(len(flux_table)):
    for j in range(len(flux_table[i])):
        print(f"Row {i}, Col {j}: {flux_table[i][j]}")
```

This is more verbose but gives you the position of each element, which is useful if you need to modify values based on their location.

## Practical Example: Source Catalog

A common pattern is storing tabular data where each row is a source:

```python
# Each source has: [name, ra, dec, flux]
catalog = [
    ["J1225+4011", 187.7, 12.3, 245.7],
    ["J1445+3131", 221.2, 31.5, 312.5],
    ["J0958+3224", 149.5, 32.4, 198.7]
]

# Access specific source
source_1 = catalog[0]
print(f"Source: {source_1[0]}, Flux: {source_1[3]} mJy")

# Find sources with flux > 200
bright_sources = []
for source in catalog:
    if source[3] > 200:
        bright_sources.append(source[0])
print(bright_sources)  # Output: ['J1225+4011', 'J1445+3131']

# Extract all fluxes (column 3)
all_fluxes = [source[3] for source in catalog]
print(all_fluxes)  # Output: [245.7, 312.5, 198.7]
```

But notice the problem: `source[3]` isn't very readable. What does index 3 mean? This is where dictionaries are better, or (much better) NumPy arrays that we'll learn later.

## Images as Nested Lists

A grayscale image can be represented as a 2D array of pixel intensities:

```python
# Simple 3x3 "image" with intensity values 0-255
image = [
    [100, 150, 200],
    [120, 180, 210],
    [110, 160, 190]
]

# Get pixel at position (1, 2)
pixel_value = image[1][2]
print(pixel_value)  # Output: 210

# Set a pixel to black (0)
image[0][0] = 0
```

Though for real images, you'll use NumPy arrays which are much faster and have more features.

## Things Worth Noting

**Irregular nested lists are allowed** but usually avoided:

```python
# Different row lengths - valid but confusing
irregular = [
    [1, 2, 3],
    [4, 5],
    [6, 7, 8, 9]
]
# This works, but accessing columns becomes tricky
```

**Memory reference trap:**

```python
# This creates three references to the SAME list
row = [0, 0, 0]
table = [row, row, row]
table[0][0] = 999
print(table)  # Output: [[999, 0, 0], [999, 0, 0], [999, 0, 0]]
# All rows changed!

# Always create new lists for each row
table = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]  # Three separate lists
```

**Accessing nonexistent indices:**

```python
table = [[1, 2], [3, 4]]
# table[2]  # IndexError! Only indices 0 and 1 exist
# table[0][5]  # IndexError! Each row only has 2 elements
```

## Try This

1. Create a 3x3 nested list of zeros, then set the diagonal elements to 1
2. Make a catalog with 3 sources, each having [name, flux, redshift], then extract all fluxes
3. Create a 2x4 table and loop through it to print each element with its row and column index
4. Given `[[1, 2, 3], [4, 5, 6], [7, 8, 9]]`, extract the second column (values 2, 5, 8)
5. Try the "wrong way" of creating a 2D list (`[[0]*3]*3`) and see what happens when you modify one element

## How This Is Typically Used in Astronomy

Nested lists can represent:
- **Multi-frequency data**: Each row is a source, each column is a frequency
- **Time series tables**: Each row is a time step, columns are measured properties
- **Images**: 2D arrays of pixel values (though NumPy arrays are much better for this)
- **Catalogs**: Each row is a source with multiple properties

However, for serious numerical work, you'll use **NumPy arrays** instead of nested lists. They're faster, use less memory, and have many more features. We'll learn those soon. Nested lists are mainly useful for small tables or when you're building up data before converting to NumPy.

## Related Lessons

**Previous**: [06_tuples_dicts_sets.md](06_tuples_dicts_sets.md) - Other data structures

**Next**: [08_if_statements.md](08_if_statements.md) - Making decisions in code

**Better alternative**: [14_numpy_basics.md](14_numpy_basics.md) - NumPy arrays for numerical 2D data
