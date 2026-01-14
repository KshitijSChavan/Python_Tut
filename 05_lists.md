# Lists

Up to now we've stored single values in variables. But astronomy involves collections: multiple flux measurements, arrays of coordinates, catalogs of sources. Python's **list** is the fundamental way to store ordered collections of data.

## Creating Lists

A list stores multiple items in order, written with square brackets. Think of it as a container that holds several values together:

```python
# Flux measurements from different observations
flux_measurements = [245.7, 189.3, 312.5, 198.7, 267.4]

# Source names
source_names = ["J1225+4011", "J1445+3131", "J0958+3224"]

print(flux_measurements)  # Output: [245.7, 189.3, 312.5, 198.7, 267.4]
print(len(flux_measurements))  # Output: 5
```

Lists can hold any type of data - numbers, strings, even other lists. Usually we keep them homogeneous (all the same type) for clarity.

## Accessing Individual Items

Here's something that takes getting used to: Python counts from 0, not 1. The first item in a list is at position 0, the second is at position 1, and so on.

```python
frequencies = [144, 323, 608, 1400]  # MHz

print(frequencies[0])  # Output: 144 (first item)
print(frequencies[1])  # Output: 323 (second item)
print(frequencies[3])  # Output: 1400 (fourth item)
```

Why start at 0? It's a computing tradition that actually makes some operations simpler. You'll get used to it.

Python also lets you count backwards from the end using negative numbers. `-1` means the last item, `-2` means second-to-last:

```python
print(frequencies[-1])  # Output: 1400 (last item)
print(frequencies[-2])  # Output: 608 (second-to-last)
```

This is really handy when you don't know how long the list is but need the last few items.

## Getting Ranges: Slicing

Often you need a portion of a list, not just one item. Slicing lets you extract ranges using `[start:end]`:

```python
measurements = [100, 150, 200, 250, 300, 350, 400]

# Get items from index 1 to 3 (not including 4)
subset = measurements[1:4]
print(subset)  # Output: [150, 200, 250]
```

The tricky part: the end index is *not* included. `[1:4]` gives you items at positions 1, 2, and 3, but stops before 4.

You can omit the start or end to slice from the beginning or to the end:

```python
first_three = measurements[:3]  # Everything before index 3
print(first_three)  # Output: [100, 150, 200]

from_third = measurements[3:]  # Everything from index 3 onwards
print(from_third)  # Output: [250, 300, 350, 400]
```

## Building Lists Gradually

You often start with an empty list and add items as you process data:

```python
catalog = []  # Empty list

# Add items one at a time
catalog.append("J1225+4011")
catalog.append("J1445+3131")
print(catalog)  # Output: ['J1225+4011', 'J1445+3131']
```

If you have multiple items to add at once, `extend()` is more efficient:

```python
new_sources = ["J0958+3224", "J1543+1528"]
catalog.extend(new_sources)
print(catalog)  # Output: ['J1225+4011', 'J1445+3131', 'J0958+3224', 'J1543+1528']
```

The difference between `append()` and `extend()`:
- `append()` adds one item (even if that item is a list)
- `extend()` adds each item from another list individually

## Changing and Removing Items

Unlike strings, lists are **mutable** - you can change them after creation:

```python
fluxes = [245.7, 189.3, 312.5]

# Fix an incorrect measurement
fluxes[1] = 195.3
print(fluxes)  # Output: [245.7, 195.3, 312.5]
```

For removing items, `remove()` deletes by value and `pop()` deletes by position:

```python
flux_list = [245.7, 189.3, 312.5, 198.7]

# Remove a specific value
flux_list.remove(312.5)
print(flux_list)  # Output: [245.7, 189.3, 198.7]

# Remove by position and get the value back
removed = flux_list.pop(1)  # Remove item at index 1
print(removed)  # Output: 189.3
print(flux_list)  # Output: [245.7, 198.7]
```

## Sorting

Sorting is straightforward but watch out: `sort()` modifies the list in place and returns nothing:

```python
flux_measurements = [312.5, 245.7, 189.3, 267.4]

flux_measurements.sort()  # Modifies the list
print(flux_measurements)  # Output: [189.3, 245.7, 267.4, 312.5]

# For descending order
flux_measurements.sort(reverse=True)
print(flux_measurements)  # Output: [312.5, 267.4, 245.7, 189.3]
```

If you want to keep the original list unchanged, use `sorted()` which creates a new sorted list:

```python
original = [312.5, 245.7, 189.3]
sorted_copy = sorted(original)
print(original)  # Output: [312.5, 245.7, 189.3] (unchanged)
print(sorted_copy)  # Output: [189.3, 245.7, 312.5]
```

## List Comprehensions

This is a compact way to create lists. Instead of writing a loop, you can do it in one line:

```python
# Create a list of squares
squares = [i ** 2 for i in range(5)]
print(squares)  # Output: [0, 1, 4, 9, 16]

# With a condition: only bright sources
flux_data = [245.7, 189.3, 312.5, 198.7, 267.4]
bright_sources = [f for f in flux_data if f > 200]
print(bright_sources)  # Output: [245.7, 312.5, 267.4]
```

Read it as: "make a list of `f` for each `f` in `flux_data` if `f > 200`". It's very readable once you get used to it.

## Things Worth Noting

**Lists are mutable, which can surprise you:**
```python
list1 = [1, 2, 3]
list2 = list1  # list2 points to the same list
list1.append(4)
print(list2)  # Output: [1, 2, 3, 4] (it changed too!)
```
If you want a separate copy, use `list1.copy()` or `list1[:]`.

**append() vs extend():**
```python
list1 = [1, 2]
list1.append([3, 4])
print(list1)  # Output: [1, 2, [3, 4]] (added the list as one item)

list2 = [1, 2]
list2.extend([3, 4])
print(list2)  # Output: [1, 2, 3, 4] (added each item)
```

**Checking membership:**
```python
sources = ["J1225+4011", "J1445+3131"]
print("J1225+4011" in sources)  # Output: True
```

## Try This

1. Create a list of 5 redshifts and print the first and last values
2. Take the list `[245.7, 189.3, 312.5, 198.7]` and extract the middle two values using slicing
3. Create an empty list, add three source names using `append()`, then remove the second one
4. Use a list comprehension to create a list of squares for numbers 1 to 10
5. Sort the list `[0.8, 1.2, 0.5, 1.7, 0.3]` in descending order

## How This Is Typically Used in Astronomy

Lists store measurement arrays (flux at different epochs), source catalogs, coordinate lists for multiple sources, quality flags, and observation metadata. When you read a source catalog, each column typically becomes a list that you can process.

## Related Lessons

**Previous**: [04_booleans_and_comparisons.md](04_booleans_and_comparisons.md) - True/False logic

**Next**: [06_tuples_dicts_sets.md](06_tuples_dicts_sets.md) - Other data structures

**You'll use this in**: [09_for_loops.md](09_for_loops.md) - Iterating through lists
