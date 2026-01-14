# Tuples, Dictionaries, and Sets

Lists are the most common way to store collections, but Python has three other useful data structures. Each has specific strengths: tuples for data that shouldn't change, dictionaries for labeled data, and sets for unique collections.

## Tuples: Immutable Lists

A **tuple** is like a list, but once created, you can't change it. Use parentheses instead of square brackets:

```python
# Coordinates of a source (RA, Dec)
position = (187.7056, 12.3911)

# Observation metadata
observation = ("LoTSS", 144.0, "2023-04-15")

print(position)  # Output: (187.7056, 12.3911)
print(position[0])  # Output: 187.7056 (access like a list)
```

Why use tuples? They're for data that represents a fixed structure - like coordinates always being (RA, Dec), or RGB colors always being (R, G, B). The immutability protects you from accidentally changing values that should stay constant.

You can unpack tuples into separate variables:

```python
ra, dec = position
print(f"RA: {ra}, Dec: {dec}")  # Output: RA: 187.7056, Dec: 12.3911

# Swapping variables uses tuples internally
a, b = 10, 20
a, b = b, a  # This creates temporary tuple (b, a)
```

Tuples can't be modified:

```python
position = (187.7, 12.3)
# position[0] = 188.0  # TypeError! Can't modify tuples
```

If you need to change values, use a list. If the data structure is fixed, use a tuple.

## Dictionaries: Labeled Data

A **dictionary** stores key-value pairs. Instead of accessing items by position (like lists), you access them by name:

```python
# Source properties with labels
source = {
    "name": "J1225+4011",
    "flux_mJy": 245.7,
    "redshift": 1.42,
    "detected": True
}

# Access by key
print(source["name"])  # Output: J1225+4011
print(source["flux_mJy"])  # Output: 245.7

# Add new key-value pair
source["spectral_index"] = -0.75

# Modify existing value
source["flux_mJy"] = 250.3
```

This is much clearer than a list where you'd have to remember "position 0 is name, position 1 is flux..." Dictionaries make your data self-documenting.

Check if a key exists:

```python
if "redshift" in source:
    print(f"Redshift: {source['redshift']}")  # Output: Redshift: 1.42

# Get with default value if key doesn't exist
size = source.get("size_arcsec", 0)  # Returns 0 if key not found
print(size)  # Output: 0
```

Loop through dictionaries:

```python
# Get all keys
for key in source.keys():
    print(key)

# Get all values
for value in source.values():
    print(value)

# Get both keys and values
for key, value in source.items():
    print(f"{key}: {value}")
```

Dictionaries are perfect for:
- Source catalogs where each source has multiple properties
- Configuration settings
- Lookup tables (e.g., telescope name → frequency)
- Grouping related data

## Sets: Unique Collections

A **set** is an unordered collection of unique items. Use curly braces:

```python
# Quality flags from observations
flags = {0, 1, 0, 0, 2, 0, 1, 0}  # Duplicates automatically removed
print(flags)  # Output: {0, 1, 2} (order may vary)

# Create from a list
source_list = ["J1225+4011", "J1445+3131", "J1225+4011"]  # Duplicate!
unique_sources = set(source_list)
print(unique_sources)  # Output: {'J1225+4011', 'J1445+3131'}
```

Sets are useful when you only care about presence/absence, not order or count:

```python
# Which sources appear in both catalogs?
catalog_A = {"J1225+4011", "J1445+3131", "J0958+3224"}
catalog_B = {"J1445+3131", "J0958+3224", "J1543+1528"}

# Intersection (in both)
common = catalog_A & catalog_B
print(common)  # Output: {'J1445+3131', 'J0958+3224'}

# Union (in either)
all_sources = catalog_A | catalog_B
print(all_sources)  # Output: {'J1225+4011', 'J1445+3131', 'J0958+3224', 'J1543+1528'}

# Difference (in A but not B)
only_in_A = catalog_A - catalog_B
print(only_in_A)  # Output: {'J1225+4011'}
```

Add and remove items:

```python
sources = {"J1225+4011", "J1445+3131"}
sources.add("J0958+3224")
sources.remove("J1445+3131")
print(sources)  # Output: {'J1225+4011', 'J0958+3224'}
```

Sets are fast for membership checking:

```python
large_catalog = set(range(1000000))  # Million items
print(500000 in large_catalog)  # Output: True (very fast!)
```

## Choosing the Right Structure

**Use a list when:**
- Order matters
- You need to access items by position
- Duplicates are meaningful (e.g., repeated measurements)

**Use a tuple when:**
- Data structure is fixed (coordinates, RGB values)
- You want to protect data from modification
- Using as dictionary keys (lists can't be keys)

**Use a dictionary when:**
- You want to label your data
- You need to look up values by name, not position
- You're storing properties of something (source properties, settings)

**Use a set when:**
- You only care about unique values
- You need fast membership testing
- You want to do set operations (intersection, union, difference)

## Things Worth Noting

**Dictionaries maintain insertion order** (since Python 3.7):
```python
d = {"z": 3, "a": 1, "m": 2}
print(list(d.keys()))  # Output: ['z', 'a', 'm'] (order preserved)
```

**Sets are unordered** - you can't index them:
```python
s = {1, 2, 3}
# print(s[0])  # TypeError! Sets don't support indexing
```

**Dictionary keys must be immutable**:
```python
# Valid keys: strings, numbers, tuples
d = {"name": "source", 42: "answer", (1, 2): "coordinate"}

# Invalid key: list (it's mutable)
# d = {[1, 2]: "value"}  # TypeError!
```

## Try This

1. Create a tuple of three flux measurements and unpack them into separate variables
2. Make a dictionary for a source with keys: "name", "ra", "dec", "flux". Then add a "redshift" key.
3. Create two sets of source names and find which sources appear in both
4. Convert a list with duplicate values `[1, 2, 2, 3, 3, 3, 4]` to a set and back to a list
5. Store three sources as dictionaries in a list (list of dictionaries)

## How This Is Typically Used in Astronomy

**Tuples**: Fixed structures like (RA, Dec), (frequency, bandwidth), or (x, y, z) coordinates.

**Dictionaries**: Source catalogs where each source is a dictionary with properties. Configuration files (telescope settings, observation parameters). Lookup tables for conversions.

**Sets**: Finding unique sources across multiple catalogs. Cross-matching (which sources appear in catalog A and B?). Removing duplicates from source lists.

When you load a FITS table or CSV, you might represent each row as a dictionary and store all rows in a list. This gives you clear, labeled access to your data.

## Related Lessons

**Previous**: [05_lists.md](05_lists.md) - Collections with order

**Next**: [07_nested_lists.md](07_nested_lists.md) - Lists within lists (2D data)

**You'll use this in**: [14_numpy_basics.md](14_numpy_basics.md) - NumPy arrays are more powerful than nested lists
