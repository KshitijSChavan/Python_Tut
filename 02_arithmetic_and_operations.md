# Arithmetic and Operations

Python handles mathematical calculations naturally. This covers basic operations, order of operations, and working with variables.

## Basic Operations

```python
# Basic arithmetic
flux_1 = 245.7
flux_2 = 189.3

sum_flux = flux_1 + flux_2  # Addition
diff_flux = flux_1 - flux_2  # Subtraction
scaled_flux = flux_1 * 2  # Multiplication
avg_flux = (flux_1 + flux_2) / 2  # Division

print(sum_flux)  # Output: 435.0
print(avg_flux)  # Output: 217.5

# Power (exponentiation)
freq_ratio = 1400 / 144
spectral_factor = freq_ratio ** (-0.7)  # Use ** for powers, not ^
print(spectral_factor)  # Output: 0.145...

# Floor division and modulo
hours = 17
block_size = 3
complete_blocks = hours // 3  # Floor division (rounds down)
leftover = hours % 3  # Modulo (remainder)
print(complete_blocks)  # Output: 5
print(leftover)  # Output: 2
```

**Key operators**:
- `+` addition, `-` subtraction, `*` multiplication, `/` division
- `**` exponentiation (power)
- `//` floor division (rounds down)
- `%` modulo (remainder)

## Order of Operations

Python follows standard precedence (PEMDAS):
1. Parentheses `()`
2. Exponentiation `**`
3. Multiplication/Division `*`, `/`, `//`, `%`
4. Addition/Subtraction `+`, `-`

```python
result = 2 + 3 * 4
print(result)  # Output: 14 (not 20, because 3*4 happens first)

result = (2 + 3) * 4
print(result)  # Output: 20 (parentheses first)

# For complex formulas, use parentheses for clarity
luminosity = 4 * pi * (distance ** 2) * flux * ((1 + z) ** (1 - alpha))
```

## Shorthand Operators

```python
flux = 100.0
flux = flux + 20  # Add 20
flux += 20  # Shorthand for above
print(flux)  # Output: 140.0

# Other shorthand operators
flux -= 10  # flux = flux - 10
flux *= 1.5  # flux = flux * 1.5
flux /= 2  # flux = flux / 2
```

## Division Behavior

```python
# Division always gives float in Python 3
result = 7 / 2
print(result)  # Output: 3.5

# Floor division gives integer result
result = 7 // 2
print(result)  # Output: 3

# Even with floats
result = 7.8 // 2.0
print(result)  # Output: 3.0
```

## Working with Scientific Notation

```python
# Large/small numbers
speed_of_light = 2.998e8  # 2.998 × 10^8 m/s
planck_constant = 6.626e-34  # 6.626 × 10^-34 J·s

distance_m = 3.086e16  # 1 parsec in meters
```

## Things Worth Noting

**Integer Division in Python 2 vs 3**: In Python 3, `/` always gives float. In old Python 2 code, `7/2` would give `3`.

**Floating Point Precision**: Computers can't represent all decimals exactly:
```python
result = 0.1 + 0.2
print(result)  # Output: 0.30000000000000004 (not exactly 0.3!)
```
This is normal and usually doesn't matter for astronomy.

**Negative Exponents**: Use parentheses for clarity:
```python
result = 2 ** -3  # Works
result = 2 ** (-3)  # Clearer
```

## Try This

1. Calculate spectral index: α = log(S₁/S₂) / log(ν₁/ν₂) with S₁=245.7, S₂=18.3, ν₁=144, ν₂=1400 (you'll need `import math` and `math.log()`)
2. Calculate luminosity distance: D_L = (c/H₀) × z with c=299792.458 km/s, H₀=70 km/s/Mpc, z=0.5
3. Evaluate: 2 + 3 * 4 ** 2 - 5 (predict first, then check)

## How This Is Typically Used in Astronomy

Every calculation uses these operations:
- Flux ratios and spectral indices (division, logarithms)
- Distance calculations (multiplication, powers)
- Unit conversions (multiplication, division)
- Statistical weights (division, square roots via `** 0.5`)
- K-corrections (powers of (1+z))

## Related Lessons

**Previous**: [01_variables_and_types.md](01_variables_and_types.md) - Storing values

**Next**: [03_strings_and_operations.md](03_strings_and_operations.md) - Working with text

**You'll use this in**: [12_statistics_from_scratch.md](12_statistics_from_scratch.md) - Calculating statistical measures
