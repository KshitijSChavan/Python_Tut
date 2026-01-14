# Astronomy Calculations

Astronomy has its own set of calculations - unit conversions, coordinate transformations, distance measures, magnitude systems. Let's implement some common ones. Later you'll use libraries like `astropy` for this, but understanding the mechanics helps you catch errors and know what the libraries are actually doing.

## Unit Conversions

Radio astronomy uses various flux density units. Converting between them is essential:

```python
def mJy_to_Jy(flux_mJy):
    """Convert flux from milliJansky to Jansky."""
    return flux_mJy / 1000.0

def Jy_to_mJy(flux_Jy):
    """Convert flux from Jansky to milliJansky."""
    return flux_Jy * 1000.0

# Usage
flux_mJy = 245.7
flux_Jy = mJy_to_Jy(flux_mJy)
print(f"{flux_mJy} mJy = {flux_Jy:.4f} Jy")
```

For luminosity calculations, you might need watts per hertz:

```python
def Jy_to_W_Hz(flux_Jy):
    """Convert Jansky to watts per hertz per square meter."""
    return flux_Jy * 1e-26  # 1 Jy = 10^-26 W Hz^-1 m^-2
```

## Frequency and Wavelength

Radio observations are described by frequency or wavelength. Converting between them uses c = λν:

```python
def freq_to_wavelength(freq_MHz):
    """
    Convert frequency to wavelength.
    
    Parameters:
        freq_MHz: frequency in MHz
    
    Returns:
        wavelength in meters
    """
    speed_of_light = 2.998e8  # m/s
    freq_Hz = freq_MHz * 1e6  # Convert MHz to Hz
    wavelength_m = speed_of_light / freq_Hz
    return wavelength_m

# LOFAR frequency
freq = 144  # MHz
wavelength = freq_to_wavelength(freq)
print(f"{freq} MHz = {wavelength:.2f} m")  # Output: 144 MHz = 2.08 m
```

And the reverse:

```python
def wavelength_to_freq(wavelength_m):
    """Convert wavelength to frequency in MHz."""
    speed_of_light = 2.998e8  # m/s
    freq_Hz = speed_of_light / wavelength_m
    freq_MHz = freq_Hz / 1e6
    return freq_MHz
```

## Spectral Index

The spectral index α describes how flux changes with frequency: S ∝ ν^α. Calculate it from two flux measurements:

```python
import math

def calculate_spectral_index(flux1, freq1, flux2, freq2):
    """
    Calculate spectral index from two flux measurements.
    
    Parameters:
        flux1, flux2: flux densities in same units
        freq1, freq2: frequencies in same units
    
    Returns:
        spectral index α where S ∝ ν^α
    """
    alpha = math.log(flux1 / flux2) / math.log(freq1 / freq2)
    return alpha

# LoTSS at 144 MHz, NVSS at 1400 MHz
flux_144 = 245.7  # mJy
flux_1400 = 18.3  # mJy

alpha = calculate_spectral_index(flux_144, 144, flux_1400, 1400)
print(f"Spectral index: α = {alpha:.3f}")
```

Steep spectrum sources (α < -0.5) are often older radio galaxies. Flat/inverted spectrum (α > -0.5) often indicates AGN cores.

## Flux at Different Frequencies

Given flux at one frequency and a spectral index, predict flux at another frequency:

```python
def extrapolate_flux(flux_ref, freq_ref, freq_target, spectral_index):
    """
    Extrapolate flux to a different frequency.
    
    Parameters:
        flux_ref: flux at reference frequency
        freq_ref: reference frequency
        freq_target: target frequency
        spectral_index: α where S ∝ ν^α
    
    Returns:
        flux at target frequency
    """
    flux_target = flux_ref * (freq_target / freq_ref) ** spectral_index
    return flux_target

# Predict GMRT 323 MHz flux from LoTSS 144 MHz
flux_144 = 245.7  # mJy
alpha = -0.75
flux_323 = extrapolate_flux(flux_144, 144, 323, alpha)
print(f"Predicted flux at 323 MHz: {flux_323:.2f} mJy")
```

## Angular Size and Physical Size

Convert angular size (arcseconds) to physical size (kiloparsecs) given distance:

```python
def angular_to_physical_size(angular_size_arcsec, distance_Mpc):
    """
    Convert angular size to physical size.
    
    Parameters:
        angular_size_arcsec: angular size in arcseconds
        distance_Mpc: distance in megaparsecs
    
    Returns:
        physical size in kiloparsecs
    """
    # Small angle approximation: θ = size / distance
    # Convert arcsec to radians: 1 arcsec = 4.848e-6 radians
    angular_size_rad = angular_size_arcsec * 4.848e-6
    
    # Convert Mpc to kpc
    distance_kpc = distance_Mpc * 1000
    
    # Physical size
    physical_size_kpc = angular_size_rad * distance_kpc
    return physical_size_kpc

# 45 arcsec source at 100 Mpc
size_arcsec = 45
distance = 100  # Mpc
physical_size = angular_to_physical_size(size_arcsec, distance)
print(f"{size_arcsec} arcsec at {distance} Mpc = {physical_size:.1f} kpc")
```

This uses the small angle approximation, valid for sizes much smaller than the distance.

## Luminosity from Flux

Convert observed flux to luminosity (intrinsic power):

```python
def flux_to_luminosity(flux_Jy, distance_Mpc, frequency_MHz):
    """
    Calculate luminosity from flux density.
    
    Parameters:
        flux_Jy: flux density in Jansky
        distance_Mpc: luminosity distance in megaparsecs
        frequency_MHz: observing frequency in MHz
    
    Returns:
        luminosity in watts per hertz
    """
    # Convert flux to W Hz^-1 m^-2
    flux_SI = flux_Jy * 1e-26
    
    # Convert distance to meters
    Mpc_to_m = 3.086e22
    distance_m = distance_Mpc * Mpc_to_m
    
    # Luminosity = flux × 4π × distance^2
    luminosity = flux_SI * 4 * 3.14159 * (distance_m ** 2)
    return luminosity

flux = 0.2457  # Jy (245.7 mJy)
distance = 1000  # Mpc
freq = 144  # MHz

L = flux_to_luminosity(flux, distance, freq)
print(f"Luminosity: {L:.3e} W/Hz")
```

## Redshift Conversions

For cosmological sources, observed frequency differs from emitted:

```python
def observed_to_rest_freq(freq_obs_MHz, redshift):
    """Convert observed frequency to rest-frame frequency."""
    freq_rest = freq_obs_MHz * (1 + redshift)
    return freq_rest

def rest_to_observed_freq(freq_rest_MHz, redshift):
    """Convert rest-frame frequency to observed frequency."""
    freq_obs = freq_rest_MHz / (1 + redshift)
    return freq_obs

# Source at z=1.5 observed at 144 MHz
z = 1.5
freq_obs = 144
freq_rest = observed_to_rest_freq(freq_obs, z)
print(f"Observed {freq_obs} MHz → Rest-frame {freq_rest:.0f} MHz at z={z}")
```

## Coordinate Conversions

Convert between decimal degrees and sexagesimal (hours:minutes:seconds):

```python
def deg_to_hms(deg):
    """
    Convert RA in decimal degrees to hours:minutes:seconds.
    
    Parameters:
        deg: RA in decimal degrees (0-360)
    
    Returns:
        tuple: (hours, minutes, seconds)
    """
    hours_decimal = deg / 15.0  # 360 deg = 24 hours
    hours = int(hours_decimal)
    minutes_decimal = (hours_decimal - hours) * 60
    minutes = int(minutes_decimal)
    seconds = (minutes_decimal - minutes) * 60
    return hours, minutes, seconds

# J1225+4011 RA
ra_deg = 186.3240
h, m, s = deg_to_hms(ra_deg)
print(f"RA: {ra_deg}° = {h:02d}h {m:02d}m {s:05.2f}s")
```

## Things Worth Noting

**Unit consistency is critical.** Always check units in your calculations. Mixing MHz with Hz or Mpc with kpc creates errors that are hard to debug. Include units in variable names: `distance_Mpc`, `flux_Jy`.

**Significant figures matter.** Don't report results with more precision than your input data justifies. If your flux measurement is 245.7 ± 10 mJy, reporting luminosity to 10 decimal places is meaningless.

**Cosmology assumptions:** The luminosity distance depends on cosmology (H₀, Ωₘ, ΩΛ). These simple formulas assume flat ΛCDM with H₀=70 km/s/Mpc. For precise work, use `astropy.cosmology`.

**Small angle approximation:** The angular-to-physical size conversion assumes the source is much smaller than the distance. For nearby extended sources, use proper angular diameter distance calculations.

**Spectral index is not always constant.** Some sources have curved spectra, where α changes with frequency. These simple linear fits in log-space are approximations.

## Try This

1. Write a function to convert flux from Jy to erg/s/cm²/Hz
2. Calculate the spectral index for a source with flux 300 mJy at 150 MHz and 50 mJy at 1400 MHz
3. Convert Dec = 40.1892° to degrees:arcminutes:arcseconds
4. Calculate the physical size of a 2 arcmin source at redshift z=0.5 (assume D≈2000 Mpc)
5. Write a function that converts frequency to wavelength and returns both meters and centimeters

## How This Is Typically Used in Astronomy

Every observation involves unit conversions (mJy ↔ Jy ↔ W/Hz), spectral index calculations for classification, coordinate transformations for catalog matching, luminosity calculations for physical interpretation, and angular-to-physical size conversions for morphology studies.

These calculations appear in every radio astronomy paper.

## Related Lessons

**Previous**: [17_matplotlib_advanced.md](17_matplotlib_advanced.md) - Visualizing results

**Next**: [19_scipy_essentials.md](19_scipy_essentials.md) - Scientific computing tools

**Better approach**: Later, use `astropy.units` and `astropy.coordinates` for automatic unit handling
