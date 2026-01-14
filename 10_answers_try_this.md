# Answers for Try This Section in 10 - Functions

## Question 1
**Original question:** Write a function that converts flux from mJy to Jy (divide by 1000)

```python
def mJy_to_Jy(flux_mJy):
    """Convert flux from milliJansky to Jansky."""
    return flux_mJy / 1000.0

# Test
flux_mJy = 245.7
flux_Jy = mJy_to_Jy(flux_mJy)
print(f"{flux_mJy} mJy = {flux_Jy:.4f} Jy")
# Output: 245.7 mJy = 0.2457 Jy

# Test with multiple values
test_fluxes = [245.7, 189.3, 312.5, 1000.0]
for f in test_fluxes:
    print(f"{f} mJy = {mJy_to_Jy(f):.4f} Jy")
```

## Question 2
**Original question:** Write a function that takes RA and Dec and returns them as a tuple

```python
def get_coordinates(ra, dec):
    """
    Package RA and Dec into a tuple.
    
    Parameters:
        ra: Right Ascension in degrees
        dec: Declination in degrees
    
    Returns:
        tuple: (ra, dec)
    """
    return (ra, dec)

# Test
ra = 187.7056
dec = 12.3911
coords = get_coordinates(ra, dec)
print(f"Coordinates: {coords}")  # Output: Coordinates: (187.7056, 12.3911)

# Unpack the result
ra_out, dec_out = get_coordinates(221.2134, 31.5203)
print(f"RA: {ra_out}, Dec: {dec_out}")
# Output: RA: 221.2134, Dec: 31.5203
```

## Question 3
**Original question:** Create a function that checks if a redshift is in range 0.5-2.0 (return True/False)

```python
def is_in_redshift_range(z):
    """
    Check if redshift is in target range.
    
    Parameters:
        z: redshift value
    
    Returns:
        bool: True if 0.5 <= z <= 2.0, False otherwise
    """
    return 0.5 <= z <= 2.0

# Test
test_redshifts = [0.3, 0.8, 1.5, 2.5]
for z in test_redshifts:
    result = is_in_redshift_range(z)
    status = "✓" if result else "✗"
    print(f"{status} z={z}: {result}")
# Output:
# ✗ z=0.3: False
# ✓ z=0.8: True
# ✓ z=1.5: True
# ✗ z=2.5: False
```

## Question 4
**Original question:** Write a function with a default parameter for the detection threshold (default=100)

```python
def is_detected(flux, threshold=100):
    """
    Check if source is detected above threshold.
    
    Parameters:
        flux: measured flux in mJy
        threshold: detection threshold in mJy (default: 100)
    
    Returns:
        bool: True if flux > threshold
    """
    return flux > threshold

# Test with default threshold
print(is_detected(245.7))  # Output: True (uses default threshold=100)
print(is_detected(89.3))   # Output: False

# Test with custom threshold
print(is_detected(245.7, threshold=200))  # Output: True
print(is_detected(150.0, threshold=200))  # Output: False

# More complete example
fluxes = [245.7, 89.3, 312.5, 45.2, 198.7]
print(f"Detected at 100 mJy: {sum(is_detected(f) for f in fluxes)} sources")
print(f"Detected at 200 mJy: {sum(is_detected(f, 200) for f in fluxes)} sources")
# Output:
# Detected at 100 mJy: 3 sources
# Detected at 200 mJy: 2 sources
```

## Question 5
**Original question:** Add a docstring to one of your functions explaining what it does

```python
def calculate_spectral_index(flux1, freq1, flux2, freq2):
    """
    Calculate the spectral index from two flux measurements.
    
    The spectral index α describes how flux changes with frequency
    according to the power law: S ∝ ν^α
    
    Parameters:
        flux1 (float): Flux at first frequency (any units)
        freq1 (float): First frequency (any units)
        flux2 (float): Flux at second frequency (same units as flux1)
        freq2 (float): Second frequency (same units as freq1)
    
    Returns:
        float: Spectral index α
    
    Example:
        >>> alpha = calculate_spectral_index(245.7, 144, 18.3, 1400)
        >>> print(f"Spectral index: {alpha:.3f}")
        Spectral index: -1.112
    
    Notes:
        - Steep spectrum sources have α < -0.5
        - Flat spectrum sources have α > -0.5
        - Inverted spectrum sources have α > 0
    """
    import math
    return math.log(flux1 / flux2) / math.log(freq1 / freq2)

# Test the function
alpha = calculate_spectral_index(245.7, 144, 18.3, 1400)
print(f"Spectral index: {alpha:.3f}")

# Access the docstring
print("\nDocstring:")
print(calculate_spectral_index.__doc__)
```
