# Answers for Try This Section in 15 - NumPy Data Loading and Analysis

## Question 1
**Original question:** Create a text file with 5 flux values and load it with NumPy

```python
import numpy as np

# Create file
flux_values = [245.7, 189.3, 312.5, 198.7, 267.4]
np.savetxt('flux_data.txt', flux_values)

# Load it back
loaded_flux = np.loadtxt('flux_data.txt')
print(f"Loaded flux values: {loaded_flux}")
# Output: Loaded flux values: [245.7 189.3 312.5 198.7 267.4]
```

## Question 2
**Original question:** Load a multi-column file and calculate the mean of one column

```python
import numpy as np

# Create sample file
data = np.array([
    [187.7, 12.3, 245.7],
    [221.2, 31.5, 312.5],
    [149.5, 32.4, 198.7]
])
np.savetxt('catalog.txt', data)

# Load
loaded_data = np.loadtxt('catalog.txt')

# Mean of third column (flux)
flux_column = loaded_data[:, 2]
mean_flux = np.mean(flux_column)
print(f"Mean flux: {mean_flux:.2f} mJy")
# Output: Mean flux: 252.30 mJy
```

## Question 3
**Original question:** Load data and filter to keep only fluxes > 200

```python
import numpy as np

# Load data
flux = np.loadtxt('flux_data.txt')

# Filter
bright_flux = flux[flux > 200]
print(f"Original: {len(flux)} values")
print(f"Bright (>200): {len(bright_flux)} values")
print(f"Bright sources: {bright_flux}")
```

## Question 4
**Original question:** Load data with NaN values and remove them

```python
import numpy as np

# Create file with missing data
with open('data_with_nan.txt', 'w') as f:
    f.write("245.7\nnan\n312.5\n198.7\nnan\n267.4\n")

# Load
data = np.genfromtxt('data_with_nan.txt')
print(f"Loaded: {data}")

# Remove NaN
clean_data = data[~np.isnan(data)]
print(f"Clean data: {clean_data}")
print(f"Removed {np.sum(np.isnan(data))} NaN values")
```

## Question 5
**Original question:** Calculate the 25th, 50th (median), and 75th percentiles of loaded data

```python
import numpy as np

flux = np.array([245.7, 189.3, 312.5, 198.7, 267.4, 234.1, 289.6])

q25 = np.percentile(flux, 25)
q50 = np.percentile(flux, 50)  # median
q75 = np.percentile(flux, 75)

print(f"Q1 (25th percentile): {q25:.2f} mJy")
print(f"Q2 (median): {q50:.2f} mJy")
print(f"Q3 (75th percentile): {q75:.2f} mJy")
print(f"IQR: {q75 - q25:.2f} mJy")
```
EOF
