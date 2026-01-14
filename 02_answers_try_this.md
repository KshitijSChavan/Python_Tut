# Answers for Try This Section in 02 - Arithmetic and Operations

## Question 1
Calculate spectral index: α = log(S₁/S₂) / log(ν₁/ν₂) with S₁=245.7, S₂=18.3, ν₁=144, ν₂=1400

```python
import math

S1 = 245.7
S2 = 18.3
nu1 = 144
nu2 = 1400

flux_ratio = S1 / S2
freq_ratio = nu1 / nu2

spectral_index = math.log(flux_ratio) / math.log(freq_ratio)
print(f"Spectral index: {spectral_index:.3f}")  # Output: Spectral index: -1.112
```

## Question 2
Calculate luminosity distance: D_L = (c/H₀) × z with c=299792.458 km/s, H₀=70 km/s/Mpc, z=0.5

```python
c = 299792.458  # km/s
H0 = 70  # km/s/Mpc
z = 0.5

D_L = (c / H0) * z
print(f"Luminosity distance: {D_L:.2f} Mpc")  # Output: Luminosity distance: 2141.37 Mpc
```

## Question 3
Evaluate: 2 + 3 * 4 ** 2 - 5

```python
result = 2 + 3 * 4 ** 2 - 5
print(result)  # Output: 45

# Step by step:
# 4 ** 2 = 16
# 3 * 16 = 48
# 2 + 48 = 50
# 50 - 5 = 45
```
