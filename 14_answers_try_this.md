# Answers for Try This Section in 14 - NumPy Basics

## Question 1
**Original question:** Create an array of frequencies from 100 to 1000 MHz in steps of 50

```python
import numpy as np

# Using arange
frequencies = np.arange(100, 1001, 50)  # 1001 because stop is not included
print(frequencies)
# Output: [ 100  150  200  250  300  350  400  450  500  550  600  650  700  750  800  850  900  950 1000]

# Alternative using linspace if you want exact number of points
freq_alt = np.linspace(100, 1000, 19)  # 19 points from 100 to 1000
print(freq_alt)
```

## Question 2
**Original question:** Convert an array of fluxes in mJy to Jy

```python
import numpy as np

flux_mJy = np.array([245.7, 189.3, 312.5, 198.7, 267.4])

# Convert mJy to Jy
flux_Jy = flux_mJy / 1000

print(f"mJy: {flux_mJy}")
print(f"Jy: {flux_Jy}")
# Output: Jy: [0.2457 0.1893 0.3125 0.1987 0.2674]
```

## Question 3
**Original question:** Filter an array to keep only values between 100 and 300

```python
import numpy as np

flux = np.array([245.7, 89.3, 312.5, 145.2, 267.4, 350.1, 178.9])

# Filter
medium_flux = flux[(flux >= 100) & (flux <= 300)]

print(f"Original: {flux}")
print(f"Between 100-300: {medium_flux}")
# Output: Between 100-300: [245.7 145.2 267.4 178.9]
```

## Question 4
**Original question:** Calculate mean and std of `[245.7, 189.3, 312.5, 198.7, 267.4]` using NumPy

```python
import numpy as np

flux = np.array([245.7, 189.3, 312.5, 198.7, 267.4])

mean = np.mean(flux)
std = np.std(flux, ddof=1)  # Sample std dev

print(f"Mean: {mean:.2f} mJy")
print(f"Std dev: {std:.2f} mJy")
# Output:
# Mean: 242.72 mJy
# Std dev: 43.76 mJy
```

## Question 5
**Original question:** Create a 3x3 array of ones, then set the diagonal to zeros

```python
import numpy as np

# Create 3x3 array of ones
arr = np.ones((3, 3))
print("Initial array:")
print(arr)

# Set diagonal to zeros
for i in range(3):
    arr[i, i] = 0

print("\nWith diagonal as zeros:")
print(arr)
# Output:
# [[0. 1. 1.]
#  [1. 0. 1.]
#  [1. 1. 0.]]

# Alternative using numpy function
arr = np.ones((3, 3))
np.fill_diagonal(arr, 0)
print("\nUsing fill_diagonal:")
print(arr)
```
EOF
cat /home/claude/14_answers_try_this.md
Output

# Answers for Try This Section in 14 - NumPy Basics

## Question 1
**Original question:** Create an array of frequencies from 100 to 1000 MHz in steps of 50

```python
import numpy as np

# Using arange
frequencies = np.arange(100, 1001, 50)  # 1001 because stop is not included
print(frequencies)
# Output: [ 100  150  200  250  300  350  400  450  500  550  600  650  700  750  800  850  900  950 1000]

# Alternative using linspace if you want exact number of points
freq_alt = np.linspace(100, 1000, 19)  # 19 points from 100 to 1000
print(freq_alt)
```

## Question 2
**Original question:** Convert an array of fluxes in mJy to Jy

```python
import numpy as np

flux_mJy = np.array([245.7, 189.3, 312.5, 198.7, 267.4])

# Convert mJy to Jy
flux_Jy = flux_mJy / 1000

print(f"mJy: {flux_mJy}")
print(f"Jy: {flux_Jy}")
# Output: Jy: [0.2457 0.1893 0.3125 0.1987 0.2674]
```

## Question 3
**Original question:** Filter an array to keep only values between 100 and 300

```python
import numpy as np

flux = np.array([245.7, 89.3, 312.5, 145.2, 267.4, 350.1, 178.9])

# Filter
medium_flux = flux[(flux >= 100) & (flux <= 300)]

print(f"Original: {flux}")
print(f"Between 100-300: {medium_flux}")
# Output: Between 100-300: [245.7 145.2 267.4 178.9]
```

## Question 4
**Original question:** Calculate mean and std of `[245.7, 189.3, 312.5, 198.7, 267.4]` using NumPy

```python
import numpy as np

flux = np.array([245.7, 189.3, 312.5, 198.7, 267.4])

mean = np.mean(flux)
std = np.std(flux, ddof=1)  # Sample std dev

print(f"Mean: {mean:.2f} mJy")
print(f"Std dev: {std:.2f} mJy")
# Output:
# Mean: 242.72 mJy
# Std dev: 43.76 mJy
```

## Question 5
**Original question:** Create a 3x3 array of ones, then set the diagonal to zeros

```python
import numpy as np

# Create 3x3 array of ones
arr = np.ones((3, 3))
print("Initial array:")
print(arr)

# Set diagonal to zeros
for i in range(3):
    arr[i, i] = 0

print("\nWith diagonal as zeros:")
print(arr)
# Output:
# [[0. 1. 1.]
#  [1. 0. 1.]
#  [1. 1. 0.]]

# Alternative using numpy function
arr = np.ones((3, 3))
np.fill_diagonal(arr, 0)
print("\nUsing fill_diagonal:")
print(arr)
```
