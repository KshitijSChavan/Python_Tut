# Answers for Try This Section in 04 - Booleans and Comparisons

## Question 1
Check if flux 150.5 is between 100 and 200 (use chained comparison)

```python
flux = 150.5
in_range = 100 < flux < 200
print(in_range)  # Output: True

# Alternative (more explicit):
in_range = (flux > 100) and (flux < 200)
print(in_range)  # Output: True
```

## Question 2
Determine if a source with spectral_index=-0.65 and size=50 is both steep spectrum (< -0.5) and extended (> 30)

```python
spectral_index = -0.65
size = 50

is_steep = spectral_index < -0.5
is_extended = size > 30
is_steep_and_extended = is_steep and is_extended

print(f"Steep spectrum: {is_steep}")  # Output: Steep spectrum: True
print(f"Extended: {is_extended}")  # Output: Extended: True
print(f"Both: {is_steep_and_extended}")  # Output: Both: True

# Or in one line:
result = (spectral_index < -0.5) and (size > 30)
print(result)  # Output: True
```

## Question 3
Check if "LoTSS" appears in the string "LoTSS DR2 Catalog"

```python
catalog_name = "LoTSS DR2 Catalog"
has_lotss = "LoTSS" in catalog_name
print(has_lotss)  # Output: True

# Can also use directly in condition:
if "LoTSS" in catalog_name:
    print("Found LoTSS!")  # Output: Found LoTSS!
```

## Question 4
Compare three redshifts [0.5, 1.2, 0.8] and identify which is highest using comparison operators

```python
z1 = 0.5
z2 = 1.2
z3 = 0.8

# Compare pairwise
print(f"z2 > z1: {z2 > z1}")  # Output: z2 > z1: True
print(f"z2 > z3: {z2 > z3}")  # Output: z2 > z3: True
print(f"z1 > z3: {z1 > z3}")  # Output: z1 > z3: False

# Find maximum
is_z1_highest = (z1 > z2) and (z1 > z3)
is_z2_highest = (z2 > z1) and (z2 > z3)
is_z3_highest = (z3 > z1) and (z3 > z2)

print(f"z1 is highest: {is_z1_highest}")  # Output: z1 is highest: False
print(f"z2 is highest: {is_z2_highest}")  # Output: z2 is highest: True
print(f"z3 is highest: {is_z3_highest}")  # Output: z3 is highest: False

# Or use Python's built-in max():
redshifts = [0.5, 1.2, 0.8]
highest = max(redshifts)
print(f"Highest redshift: {highest}")  # Output: Highest redshift: 1.2
```
