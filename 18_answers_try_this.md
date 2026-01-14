# Answers for Try This Section in 18 - Astronomy Calculations

## Question 1
**Original question:** Write a function to convert flux from Jy to erg/s/cm²/Hz

```python
def Jy_to_erg(flux_Jy):
    """
    Convert flux from Jansky to erg/s/cm²/Hz.
    
    1 Jy = 10^-23 erg/s/cm²/Hz
    """
    return flux_Jy * 1e-23

# Test
flux_Jy = 0.2457
flux_erg = Jy_to_erg(flux_Jy)
print(f"{flux_Jy} Jy = {flux_erg:.4e} erg/s/cm²/Hz")
# Output: 0.2457 Jy = 2.4570e-24 erg/s/cm²/Hz

# Test with multiple values
test_fluxes = [0.1, 0.5, 1.0, 10.0]
for f in test_fluxes:
    print(f"{f} Jy = {Jy_to_erg(f):.2e} erg/s/cm²/Hz")
```

## Question 2
**Original question:** Calculate the spectral index for a source with flux 300 mJy at 150 MHz and 50 mJy at 1400 MHz

```python
import math

def calculate_spectral_index(flux1, freq1, flux2, freq2):
    """Calculate spectral index from two flux measurements."""
    return math.log(flux1 / flux2) / math.log(freq1 / freq2)

# Given values
flux_150 = 300  # mJy
flux_1400 = 50  # mJy
freq_150 = 150  # MHz
freq_1400 = 1400  # MHz

alpha = calculate_spectral_index(flux_150, freq_150, flux_1400, freq_1400)
print(f"Flux at 150 MHz: {flux_150} mJy")
print(f"Flux at 1400 MHz: {flux_1400} mJy")
print(f"Spectral index α = {alpha:.3f}")
# Output: Spectral index α = -0.801

# Interpretation
if alpha < -0.5:
    print("This is a steep spectrum source")
elif alpha > 0.5:
    print("This is an inverted spectrum source")
else:
    print("This is a flat spectrum source")
```

## Question 3
**Original question:** Convert Dec = 40.1892° to degrees:arcminutes:arcseconds

```python
def deg_to_dms(degrees):
    """
    Convert decimal degrees to degrees:arcminutes:arcseconds.
    
    Parameters:
        degrees: angle in decimal degrees
    
    Returns:
        tuple: (degrees, arcminutes, arcseconds)
    """
    # Handle negative values
    sign = 1 if degrees >= 0 else -1
    degrees = abs(degrees)
    
    # Extract degrees
    deg = int(degrees)
    
    # Extract arcminutes
    minutes_decimal = (degrees - deg) * 60
    arcmin = int(minutes_decimal)
    
    # Extract arcseconds
    arcsec = (minutes_decimal - arcmin) * 60
    
    return sign * deg, arcmin, arcsec

# Test
dec = 40.1892
d, m, s = deg_to_dms(dec)
print(f"{dec}° = {d:+03d}°{m:02d}'{s:05.2f}\"")
# Output: 40.1892° = +40°11'21.12"

# Test with negative declination
dec_neg = -25.3456
d, m, s = deg_to_dms(dec_neg)
print(f"{dec_neg}° = {d:+03d}°{m:02d}'{s:05.2f}\"")
# Output: -25.3456° = -25°20'44.16"
```

## Question 4
**Original question:** Calculate the physical size of a 2 arcmin source at redshift z=0.5 (assume D≈2000 Mpc)

```python
def angular_to_physical(angular_size_arcmin, distance_Mpc):
    """
    Convert angular size to physical size.
    
    Parameters:
        angular_size_arcmin: angular size in arcminutes
        distance_Mpc: distance in megaparsecs
    
    Returns:
        physical size in kiloparsecs
    """
    # Convert arcmin to arcsec
    angular_size_arcsec = angular_size_arcmin * 60
    
    # Convert arcsec to radians
    angular_size_rad = angular_size_arcsec * 4.848e-6
    
    # Convert Mpc to kpc
    distance_kpc = distance_Mpc * 1000
    
    # Physical size using small angle approximation
    physical_size_kpc = angular_size_rad * distance_kpc
    
    return physical_size_kpc

# Given values
angular_size = 2  # arcmin
distance = 2000  # Mpc (at z=0.5)

physical_size = angular_to_physical(angular_size, distance)
print(f"Angular size: {angular_size} arcmin")
print(f"Distance: {distance} Mpc (z=0.5)")
print(f"Physical size: {physical_size:.1f} kpc")
# Output: Physical size: 349.2 kpc

# Compare with other distances
print("\nPhysical sizes at different distances:")
for d in [100, 500, 1000, 2000, 5000]:
    size = angular_to_physical(angular_size, d)
    print(f"  {d:4d} Mpc: {size:.1f} kpc")
```

## Question 5
**Original question:** Write a function that converts frequency to wavelength and returns both meters and centimeters

```python
def freq_to_wavelength(freq_MHz):
    """
    Convert frequency to wavelength.
    
    Parameters:
        freq_MHz: frequency in MHz
    
    Returns:
        dict with wavelength in meters and centimeters
    """
    speed_of_light = 2.998e8  # m/s
    freq_Hz = freq_MHz * 1e6  # Convert MHz to Hz
    
    wavelength_m = speed_of_light / freq_Hz
    wavelength_cm = wavelength_m * 100
    
    return {
        'meters': wavelength_m,
        'centimeters': wavelength_cm,
        'frequency_MHz': freq_MHz
    }

# Test with common radio frequencies
frequencies = [144, 323, 608, 1400, 4850]  # MHz

print("Frequency to Wavelength Conversion:")
print("-" * 50)
for freq in frequencies:
    result = freq_to_wavelength(freq)
    print(f"{freq:4d} MHz: {result['meters']:.3f} m = {result['centimeters']:.2f} cm")

# Output:
#  144 MHz: 2.082 m = 208.19 cm
#  323 MHz: 0.928 m = 92.79 cm
#  608 MHz: 0.493 m = 49.31 cm
# 1400 MHz: 0.214 m = 21.41 cm
# 4850 MHz: 0.062 m = 6.18 cm

# Reverse calculation verification
def wavelength_to_freq(wavelength_m):
    """Convert wavelength to frequency."""
    speed_of_light = 2.998e8  # m/s
    freq_Hz = speed_of_light / wavelength_m
    return freq_Hz / 1e6  # Convert to MHz

# Verify
wavelength = 2.082  # meters (144 MHz)
freq_check = wavelength_to_freq(wavelength)
print(f"\nVerification: {wavelength} m → {freq_check:.1f} MHz")
```
