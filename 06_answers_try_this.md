# Answers for Try This Section in 06 - Tuples, Dictionaries, and Sets

## Question 1
**Original question:** Create a tuple of three flux measurements and unpack them into separate variables

```python
# Create tuple
flux_measurements = (245.7, 189.3, 312.5)

# Unpack into separate variables
flux1, flux2, flux3 = flux_measurements

print(f"Flux 1: {flux1} mJy")  # Output: Flux 1: 245.7 mJy
print(f"Flux 2: {flux2} mJy")  # Output: Flux 2: 189.3 mJy
print(f"Flux 3: {flux3} mJy")  # Output: Flux 3: 312.5 mJy
```

## Question 2
**Original question:** Make a dictionary for a source with keys: "name", "ra", "dec", "flux". Then add a "redshift" key.

```python
# Create dictionary
source = {
    "name": "J1225+4011",
    "ra": 187.7056,
    "dec": 12.3911,
    "flux": 245.7
}

print(f"Initial source: {source}")

# Add redshift key
source["redshift"] = 1.42

print(f"After adding redshift: {source}")
# Output: {'name': 'J1225+4011', 'ra': 187.7056, 'dec': 12.3911, 'flux': 245.7, 'redshift': 1.42}
```

## Question 3
**Original question:** Create two sets of source names and find which sources appear in both

```python
# Two catalogs
catalog_A = {"J1225+4011", "J1445+3131", "J0958+3224", "J1543+1528"}
catalog_B = {"J1445+3131", "J0958+3224", "J2134+0042", "J0823+2456"}

# Find intersection (sources in both)
common_sources = catalog_A & catalog_B
print(f"Sources in both catalogs: {common_sources}")
# Output: Sources in both catalogs: {'J1445+3131', 'J0958+3224'}

# Alternative using intersection method
common_sources_alt = catalog_A.intersection(catalog_B)
print(f"Using .intersection(): {common_sources_alt}")
```

## Question 4
**Original question:** Convert a list with duplicate values `[1, 2, 2, 3, 3, 3, 4]` to a set and back to a list

```python
# Original list with duplicates
numbers = [1, 2, 2, 3, 3, 3, 4]
print(f"Original list: {numbers}")

# Convert to set (removes duplicates)
unique_numbers = set(numbers)
print(f"As set: {unique_numbers}")  # Output: {1, 2, 3, 4} (order may vary)

# Convert back to list
unique_list = list(unique_numbers)
print(f"Back to list: {unique_list}")  # Output: [1, 2, 3, 4] (order may vary)

# Note: Order is not preserved when converting to/from set
# If you need sorted order:
sorted_unique = sorted(list(set(numbers)))
print(f"Sorted unique: {sorted_unique}")  # Output: [1, 2, 3, 4]
```

## Question 5
**Original question:** Store three sources as dictionaries in a list (list of dictionaries)

```python
# Create list of dictionaries
sources = [
    {
        "name": "J1225+4011",
        "ra": 187.7056,
        "dec": 12.3911,
        "flux": 245.7
    },
    {
        "name": "J1445+3131",
        "ra": 221.2134,
        "dec": 31.5203,
        "flux": 312.5
    },
    {
        "name": "J0958+3224",
        "ra": 149.5421,
        "dec": 32.4156,
        "flux": 198.7
    }
]

# Access data
print(f"First source name: {sources[0]['name']}")  # Output: First source name: J1225+4011
print(f"Second source flux: {sources[1]['flux']} mJy")  # Output: Second source flux: 312.5 mJy

# Loop through all sources
for source in sources:
    print(f"{source['name']}: {source['flux']} mJy at ({source['ra']}, {source['dec']})")
```
