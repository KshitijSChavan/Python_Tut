# Strings and Operations

Strings store text data like source names, catalog IDs, telescope names, and file paths. Python provides many operations for working with text.

## Creating Strings

```python
# Single or double quotes work the same
source_name = "J1225+4011"
telescope = 'LOFAR'
catalog = "LoTSS DR2"

print(source_name)  # Output: J1225+4011
```

## String Operations

```python
# Concatenation (joining)
prefix = "LoTSS_"
source_num = "12345"
catalog_id = prefix + source_num
print(catalog_id)  # Output: LoTSS_12345

# Repetition
separator = "-" * 40
print(separator)  # Output: ----------------------------------------

# Length
name = "J1225+4011"
print(len(name))  # Output: 10

# Combining strings with numbers requires conversion
filename = "obs_" + str(42) + ".fits"
print(filename)  # Output: obs_42.fits
```

## Indexing and Slicing

Strings are sequences - you can access individual characters or ranges.

```python
source_id = "ILTJ122517.77+401124.6"

# Indexing (starts at 0)
print(source_id[0])  # Output: I
print(source_id[4])  # Output: 1
print(source_id[-1])  # Output: 6 (last character)

# Slicing [start:end]
prefix = source_id[0:4]  # Characters 0,1,2,3 (not including 4)
print(prefix)  # Output: ILTJ

ra_part = source_id[4:13]  # Extract RA portion
print(ra_part)  # Output: 122517.77

# Omit start or end
first_part = source_id[:4]  # From beginning to 4
rest = source_id[4:]  # From 4 to end
```

## Common String Methods

```python
name = "lotss dr2"

# Case conversion
print(name.upper())  # Output: LOTSS DR2
print(name.lower())  # Output: lotss dr2
print(name.title())  # Output: Lotss Dr2

# Check start/end
catalog_id = "LoTSS_12345"
print(catalog_id.startswith("LoTSS"))  # Output: True
print(catalog_id.endswith("12345"))  # Output: True

# Replace text
old_id = "DR1_source_123"
new_id = old_id.replace("DR1", "DR2")
print(new_id)  # Output: DR2_source_123

# Check if substring exists
if "LoTSS" in catalog_id:
    print("This is a LoTSS source")  # Output: This is a LoTSS source
```

## Special Characters

```python
# Newline and tab
text = "Line 1\nLine 2"  # \n creates new line
print(text)
# Output:
# Line 1
# Line 2

# Quotes inside strings
text1 = "The source's flux"  # Single quote inside double quotes
text2 = 'They said "interesting"'  # Double quotes inside single quotes
text3 = "They said \"interesting\""  # Or escape with backslash
```

## Things Worth Noting

**Strings Are Immutable**: Can't change characters after creation:
```python
name = "LoTSS"
# name[0] = "X"  # Error! Can't modify strings
name = "XoTSS"  # But can reassign the whole variable
```

**String vs Number**: `"123"` is text, not a number:
```python
x = "10"
y = "20"
print(x + y)  # Output: 1020 (concatenation, not addition!)

# Convert to numbers for math
print(int(x) + int(y))  # Output: 30
```

**Quotes Must Match**:
```python
good = "LoTSS"
good = 'LoTSS'
# bad = "LoTSS'  # Error!
```

## Try This

1. Create a source ID like "LoTSS_DR2_12345" and extract just "12345" using slicing
2. Take telescope name "lofar" and convert to uppercase
3. Build a filename using: prefix="catalog", number=42, extension=".fits"
4. Check if "GMRT" appears in telescope name "VLA/GMRT/LOFAR"

## How This Is Typically Used in Astronomy

Strings handle:
- Source identifiers: "ILTJ122517.77+401124.6", "J1225+4011"
- Catalog/survey names: "LoTSS DR2", "FIRST", "NVSS"
- File paths: "/data/observations/source_catalog.fits"
- Classifications: "FR-I", "FR-II", "Seyfert"
- Telescope/instrument names: "LOFAR", "GMRT", "VLA"

## Related Lessons

**Previous**: [02_arithmetic_and_operations.md](02_arithmetic_and_operations.md) - Working with numbers

**Next**: [04_booleans_and_comparisons.md](04_booleans_and_comparisons.md) - True/False logic

**You'll use this in**: [11_file_io_and_csv.md](11_file_io_and_csv.md) - Reading filenames and paths
