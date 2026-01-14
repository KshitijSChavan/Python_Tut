# Answers for Try This Section in 09 - For Loops

## Question 1
**Original question:** Loop through `[0.5, 1.2, 0.8, 1.7, 0.3]` and count how many values are greater than 1.0

```python
redshifts = [0.5, 1.2, 0.8, 1.7, 0.3]

count = 0
for z in redshifts:
    if z > 1.0:
        count += 1

print(f"Count greater than 1.0: {count}")  # Output: Count greater than 1.0: 2

# Alternative: show which values
high_z = []
for z in redshifts:
    if z > 1.0:
        high_z.append(z)
print(f"Values > 1.0: {high_z}")  # Output: Values > 1.0: [1.2, 1.7]
```

## Question 2
**Original question:** Use range() to calculate the sum of numbers from 1 to 100

```python
total = 0
for i in range(1, 101):  # 1 to 100 inclusive
    total += i

print(f"Sum of 1 to 100: {total}")  # Output: Sum of 1 to 100: 5050

# Verify with formula: n(n+1)/2
formula_result = 100 * 101 // 2
print(f"Using formula: {formula_result}")  # Output: Using formula: 5050

# Alternative one-liner
total_alt = sum(range(1, 101))
print(f"Using sum(): {total_alt}")
```

## Question 3
**Original question:** Loop through a list of fluxes and build a new list with only values > 200

```python
flux_list = [245.7, 89.3, 312.5, 145.2, 267.4, 178.9]

bright_fluxes = []
for flux in flux_list:
    if flux > 200:
        bright_fluxes.append(flux)

print(f"Original: {flux_list}")
print(f"Bright (>200): {bright_fluxes}")
# Output: Bright (>200): [245.7, 312.5, 267.4]

# Alternative: list comprehension (one line)
bright_fluxes_alt = [f for f in flux_list if f > 200]
print(f"Using list comprehension: {bright_fluxes_alt}")
```

## Question 4
**Original question:** Use enumerate() to print source names with their position numbers (starting from 1)

```python
source_names = ["J1225+4011", "J1445+3131", "J0958+3224", "J1543+1528"]

for i, name in enumerate(source_names):
    print(f"Source {i+1}: {name}")

# Output:
# Source 1: J1225+4011
# Source 2: J1445+3131
# Source 3: J0958+3224
# Source 4: J1543+1528

# Alternative: start enumerate at 1
for i, name in enumerate(source_names, start=1):
    print(f"Source {i}: {name}")
```

## Question 5
**Original question:** Use nested loops to print a 3x3 multiplication table

```python
# 3x3 multiplication table
for i in range(1, 4):
    for j in range(1, 4):
        product = i * j
        print(f"{i} × {j} = {product}")
    print()  # Blank line after each row

# Better formatted output
print("3x3 Multiplication Table:")
for i in range(1, 4):
    row = []
    for j in range(1, 4):
        row.append(i * j)
    print(row)
# Output:
# [1, 2, 3]
# [2, 4, 6]
# [3, 6, 9]

# Nicely formatted table
print("\n   1  2  3")
print("  --------")
for i in range(1, 4):
    print(f"{i} |", end="")
    for j in range(1, 4):
        print(f"{i*j:2d} ", end="")
    print()
# Output:
#    1  2  3
#   --------
# 1 | 1  2  3
# 2 | 2  4  6
# 3 | 3  6  9
```
