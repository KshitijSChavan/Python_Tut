# Answers: Lesson 03 - Strings and Operations

## Question 1
Create a source ID like "LoTSS_DR2_12345" and extract just "12345" using slicing

```python
source_id = "LoTSS_DR2_12345"
number = source_id[10:]  # Start from position 10 to end
print(number)  # Output: 12345

# Alternative - if you know the pattern:
number = source_id[-5:]  # Last 5 characters
print(number)  # Output: 12345
```

## Question 2
Take telescope name "lofar" and convert to uppercase

```python
telescope = "lofar"
telescope_upper = telescope.upper()
print(telescope_upper)  # Output: LOFAR
```

## Question 3
Build a filename using: prefix="catalog", number=42, extension=".fits"

```python
prefix = "catalog"
number = 42
extension = ".fits"

filename = prefix + "_" + str(number) + extension
print(filename)  # Output: catalog_42.fits

# Alternative using f-string:
filename = f"{prefix}_{number}{extension}"
print(filename)  # Output: catalog_42.fits
```

## Question 4
Check if "GMRT" appears in telescope name "VLA/GMRT/LOFAR"

```python
telescope_list = "VLA/GMRT/LOFAR"

if "GMRT" in telescope_list:
    print("GMRT found!")  # Output: GMRT found!

# Can also get True/False directly:
has_gmrt = "GMRT" in telescope_list
print(has_gmrt)  # Output: True
```
