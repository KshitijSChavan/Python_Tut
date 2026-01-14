# Variables and Types

Python uses **variables** to store data like flux measurements, coordinates, and source names. Each value has a **type**.

## Basic Example

```python
# Different types of variables
flux = 245.7  # Float (decimal number)
num_sources = 1547  # Integer (whole number)
source_name = "J1225+4011"  # String (text)
is_detected = True  # Boolean (True/False)

print(flux)  # Output: 245.7
print(type(flux))  # Output: <class 'float'>
```

## Four Main Types

**Integer (int)**: Whole numbers - `42`, `1547`, `-3`

**Float**: Decimal numbers - `245.7`, `1.42`, `144.0`

**String (str)**: Text in quotes - `"J1225+4011"`, `"LoTSS"`

**Boolean (bool)**: `True` or `False`

## Type Conversion

```python
# String to number
flux_string = "245.7"
flux_number = float(flux_string)

# Number to string
filename = "obs_" + str(42) + ".fits"  # Output: obs_42.fits

# Float to int (drops decimal)
int(245.7)  # Output: 245

# Check: int("hello") causes ValueError
# Check: int("42.7") causes ValueError - use int(float("42.7"))
```

## Variable Naming Rules

**Valid**: Letters, numbers, underscores. Must start with letter or underscore.
**Invalid**: Can't start with number, no spaces, no hyphens.

```python
flux_144MHz = 245.7  # Valid
# 144MHz_flux = 245.7  # Invalid - starts with number
# flux-density = 100  # Invalid - has hyphen
```

Use descriptive names: `flux_density` not `f`. Case-sensitive: `Flux` ≠ `flux`.

## Multiple Assignment

```python
# Assign multiple variables at once
ra, dec = 187.7, 12.3

# Swap values
a, b = 10, 20
a, b = b, a  # Now a=20, b=10
```

## Constants

Define constants manually or use NumPy (we'll learn this later):

```python
pi = 3.14159265359
# Or later: import numpy as np; pi = np.pi
```

## Try This

1. Create variables for RA (187.5), Dec (12.3), flux (245.7), and source name
2. Convert "1.42" to a float and multiply by 2
3. Swap two variables without a temporary variable

## How This Is Typically Used in Astronomy

Variables store measurements (flux, redshift, coordinates), identifiers (source names, catalog IDs), observational parameters (frequency, bandwidth), and flags (detection status, quality).

## Related Lessons

**Next**: [02_arithmetic_and_operations.md](02_arithmetic_and_operations.md) - Doing calculations
