# For Loops

What if you need to process 1000 sources? Or calculate something for each flux measurement? You don't want to copy-paste code 1000 times. **For loops** let you repeat code automatically.

## Looping Over Lists

The most natural way to loop is going directly through items in a list:

```python
flux_measurements = [245.7, 189.3, 312.5, 198.7, 267.4]

for flux in flux_measurements:
    print(f"Flux: {flux} mJy")
    
# Output:
# Flux: 245.7 mJy
# Flux: 189.3 mJy
# Flux: 312.5 mJy
# Flux: 198.7 mJy
# Flux: 267.4 mJy
```

Read this as: "for each flux in flux_measurements, do this." The variable `flux` takes on each value in the list, one at a time.

You can name the loop variable anything, but make it descriptive:

```python
source_names = ["J1225+4011", "J1445+3131", "J0958+3224"]

for name in source_names:
    print(f"Processing {name}")
```

## Using range() for Numbers

Sometimes you need to repeat something a specific number of times. Use `range()`:

```python
# Print numbers 0 to 4
for i in range(5):
    print(i)  # Output: 0, 1, 2, 3, 4
```

Remember `range(5)` gives you 5 numbers starting at 0. This is the pattern you'll use most: `range(n)` for n iterations.

You can also specify start and stop:

```python
for i in range(2, 6):
    print(i)  # Output: 2, 3, 4, 5
```

Or add a step:

```python
# Count by 100s
for freq in range(100, 1001, 100):
    print(f"{freq} MHz")
    
# Output: 100 MHz, 200 MHz, ..., 1000 MHz
```

## Accumulating Results

A common pattern is building up a result as you loop:

```python
flux_measurements = [245.7, 189.3, 312.5, 198.7, 267.4]

total = 0
for flux in flux_measurements:
    total = total + flux
    
mean = total / len(flux_measurements)
print(f"Mean flux: {mean:.2f} mJy")  # Output: Mean flux: 239.60 mJy
```

Start with an initial value (often 0 or an empty list), then update it in each iteration. This is called the **accumulator pattern**.

Building a filtered list:

```python
all_fluxes = [245.7, 89.3, 312.5, 45.2, 267.4]
threshold = 100

bright_sources = []
for flux in all_fluxes:
    if flux > threshold:
        bright_sources.append(flux)
        
print(bright_sources)  # Output: [245.7, 312.5, 267.4]
```

## Looping with Indices

Sometimes you need to know the position of each item:

```python
sources = ["J1225+4011", "J1445+3131", "J0958+3224"]

for i in range(len(sources)):
    print(f"Source {i+1}: {sources[i]}")
    
# Output:
# Source 1: J1225+4011
# Source 2: J1445+3131
# Source 3: J0958+3224
```

We use `i+1` because humans count from 1 but Python counts from 0.

There's a cleaner way to get both index and value - `enumerate()`:

```python
for i, source in enumerate(sources):
    print(f"Source {i+1}: {source}")
```

This is more Pythonic and avoids indexing errors.

## While Loops

While loops repeat as long as a condition is True:

```python
flux = 1000.0
year = 0

while flux > 100:
    flux = flux * 0.9  # 10% decay per year
    year = year + 1
    
print(f"Dropped below threshold after {year} years")
# Output: Dropped below threshold after 22 years
```

Be careful - if the condition never becomes False, you get an infinite loop! Your program will hang and you'll need to stop it manually (Ctrl+C in terminal).

```python
# Don't do this!
# x = 10
# while x > 0:
#     print(x)  # Forgot to decrease x - infinite loop!
```

Most of the time, for loops are safer and clearer than while loops.

## break and continue

**break** exits the loop immediately:

```python
sources = ["J1225+4011", "J1445+3131", "CORRUPT", "J0958+3224"]

for source in sources:
    if source == "CORRUPT":
        print("Found corrupted data, stopping")
        break
    print(f"Processing {source}")
    
# Output:
# Processing J1225+4011
# Processing J1445+3131
# Found corrupted data, stopping
```

Once break runs, Python exits the entire loop - it doesn't process the remaining sources.

**continue** skips to the next iteration:

```python
quality_flags = [0, 1, 0, 2, 0, 1]

for i, flag in enumerate(quality_flags):
    if flag != 0:
        continue  # Skip bad quality
    print(f"Processing source {i}")
    
# Output:
# Processing source 0
# Processing source 2
# Processing source 4
```

Continue says "skip the rest of this iteration and move to the next one."

## Nested Loops

Loops inside loops process 2D data:

```python
# Flux at 3 frequencies for 2 sources
flux_table = [
    [245.7, 189.3, 156.2],
    [312.5, 278.9, 234.1]
]

for source_fluxes in flux_table:
    for flux in source_fluxes:
        print(f"{flux:.1f}", end=" ")
    print()  # New line after each source
    
# Output:
# 245.7 189.3 156.2
# 312.5 278.9 234.1
```

The outer loop goes through sources, the inner loop goes through frequencies for each source.

## Things Worth Noting

**Don't modify a list while looping over it:**

```python
fluxes = [245.7, 89.3, 312.5, 45.2]

# BAD - modifying list during loop can cause unexpected behavior
# for flux in fluxes:
#     if flux < 100:
#         fluxes.remove(flux)  # Don't do this!

# GOOD - create a new list
bright_fluxes = [f for f in fluxes if f >= 100]
```

**Indentation matters:**

```python
for i in range(3):
    print("Inside loop")
print("Outside loop")  # This runs once after the loop finishes
```

If you forget to indent code that should be in the loop, it won't repeat.

**Loop variables persist:**

```python
for i in range(3):
    pass

print(i)  # Output: 2 (the last value)
```

The variable still exists after the loop ends, with the last value it had.

## Try This

1. Loop through `[0.5, 1.2, 0.8, 1.7, 0.3]` and count how many values are greater than 1.0
2. Use range() to calculate the sum of numbers from 1 to 100
3. Loop through a list of fluxes and build a new list with only values > 200
4. Use enumerate() to print source names with their position numbers (starting from 1)
5. Use nested loops to print a 3x3 multiplication table

## How This Is Typically Used in Astronomy

Loops process catalogs (go through each source), calculate statistics (accumulate sums, counts), filter data (build lists of sources meeting criteria), apply transformations (convert units, calculate derived quantities), and read files line by line.

## Related Lessons

**Previous**: [08_if_statements.md](08_if_statements.md) - Making decisions

**Next**: [10_functions.md](10_functions.md) - Reusable code blocks

**Uses**: [05_lists.md](05_lists.md) - The collections we loop through
