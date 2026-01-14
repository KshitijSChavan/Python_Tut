# Answers for Try This Section in 08 - If Statements and Logical Operators

## Question 1
**Original question:** Write an if-elif-else that classifies flux into "Bright" (>300), "Medium" (100-300), or "Faint" (<100)

```python
flux = 245.7  # Try different values

if flux > 300:
    classification = "Bright"
elif flux >= 100:
    classification = "Medium"
else:
    classification = "Faint"

print(f"Flux {flux} mJy is {classification}")
# Output: Flux 245.7 mJy is Medium

# Test with different values
for test_flux in [350, 150, 50]:
    if test_flux > 300:
        result = "Bright"
    elif test_flux >= 100:
        result = "Medium"
    else:
        result = "Faint"
    print(f"Flux {test_flux} mJy: {result}")
# Output:
# Flux 350 mJy: Bright
# Flux 150 mJy: Medium
# Flux 50 mJy: Faint
```

## Question 2
**Original question:** Check if a source with redshift=1.5 and flux=250 meets criteria: 1.0 < z < 2.0 AND flux > 200

```python
redshift = 1.5
flux = 250

# Check both conditions
in_sample = (1.0 < redshift < 2.0) and (flux > 200)

print(f"Redshift: {redshift}, Flux: {flux} mJy")
print(f"Meets criteria: {in_sample}")  # Output: Meets criteria: True

# More detailed check
if (1.0 < redshift < 2.0) and (flux > 200):
    print("Source included in sample")
else:
    print("Source excluded from sample")
# Output: Source included in sample

# Test edge cases
test_cases = [
    (1.5, 250, True),   # Should pass
    (0.5, 250, False),  # Redshift too low
    (1.5, 150, False),  # Flux too low
    (2.5, 300, False),  # Redshift too high
]

for z, f, expected in test_cases:
    result = (1.0 < z < 2.0) and (f > 200)
    status = "✓" if result == expected else "✗"
    print(f"{status} z={z}, flux={f}: {result}")
```

## Question 3
**Original question:** Check if a telescope name is in the list ["LOFAR", "GMRT", "VLA"]

```python
telescope = "GMRT"
telescope_list = ["LOFAR", "GMRT", "VLA"]

if telescope in telescope_list:
    print(f"{telescope} is in the list")
else:
    print(f"{telescope} is not in the list")
# Output: GMRT is in the list

# Test with different telescopes
test_telescopes = ["LOFAR", "JVLA", "ALMA", "VLA"]
for tel in test_telescopes:
    if tel in telescope_list:
        print(f"✓ {tel} is a recognized telescope")
    else:
        print(f"✗ {tel} is not in our list")
# Output:
# ✓ LOFAR is a recognized telescope
# ✗ JVLA is not in our list
# ✗ ALMA is not in our list
# ✓ VLA is a recognized telescope
```

## Question 4
**Original question:** Write code that skips processing if quality_flag is not 0

```python
quality_flag = 0
flux = 245.7

if quality_flag != 0:
    print("Bad quality - skipping source")
else:
    print(f"Good quality - processing flux: {flux} mJy")
# Output: Good quality - processing flux: 245.7 mJy

# More complete example
sources = [
    ("J1225+4011", 245.7, 0),  # Good
    ("J1445+3131", 312.5, 1),  # Bad
    ("J0958+3224", 198.7, 0),  # Good
    ("J1543+1528", 289.3, 2),  # Bad
]

processed_count = 0
for name, flux, flag in sources:
    if flag != 0:
        print(f"Skipping {name} - bad quality (flag={flag})")
        continue
    
    # Process source
    print(f"Processing {name}: {flux} mJy")
    processed_count += 1

print(f"\nProcessed {processed_count} out of {len(sources)} sources")
# Output:
# Processing J1225+4011: 245.7 mJy
# Skipping J1445+3131 - bad quality (flag=1)
# Processing J0958+3224: 198.7 mJy
# Skipping J1543+1528 - bad quality (flag=2)
# Processed 2 out of 4 sources
```
