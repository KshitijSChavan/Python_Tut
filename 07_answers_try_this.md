# Answers for Try This Section in 07 - Nested Lists

## Question 1
**Original question:** Create a 3x3 nested list of zeros, then set the diagonal elements to 1

```python
# Create 3x3 list of zeros
matrix = [[0, 0, 0],
          [0, 0, 0],
          [0, 0, 0]]

# Or using list comprehension
matrix = [[0 for j in range(3)] for i in range(3)]

print("Before:")
for row in matrix:
    print(row)

# Set diagonal to 1
matrix[0][0] = 1
matrix[1][1] = 1
matrix[2][2] = 1

print("\nAfter:")
for row in matrix:
    print(row)
# Output:
# [1, 0, 0]
# [0, 1, 0]
# [0, 0, 1]

# Or using a loop
matrix = [[0 for j in range(3)] for i in range(3)]
for i in range(3):
    matrix[i][i] = 1
```

## Question 2
**Original question:** Make a catalog with 3 sources, each having [name, flux, redshift], then extract all fluxes

```python
# Create catalog
catalog = [
    ["J1225+4011", 245.7, 1.42],
    ["J1445+3131", 312.5, 0.87],
    ["J0958+3224", 198.7, 1.15]
]

# Extract all fluxes (column 1)
fluxes = [source[1] for source in catalog]
print(f"All fluxes: {fluxes}")  # Output: All fluxes: [245.7, 312.5, 198.7]

# Alternative without list comprehension
fluxes = []
for source in catalog:
    fluxes.append(source[1])
print(f"Fluxes: {fluxes}")

# Can also extract names and redshifts
names = [source[0] for source in catalog]
redshifts = [source[2] for source in catalog]
print(f"Names: {names}")
print(f"Redshifts: {redshifts}")
```

## Question 3
**Original question:** Create a 2x4 table and loop through it to print each element with its row and column index

```python
# Create 2x4 table
table = [
    [10, 20, 30, 40],
    [50, 60, 70, 80]
]

# Loop with row and column indices
for i in range(len(table)):
    for j in range(len(table[i])):
        print(f"Row {i}, Col {j}: {table[i][j]}")

# Output:
# Row 0, Col 0: 10
# Row 0, Col 1: 20
# Row 0, Col 2: 30
# Row 0, Col 3: 40
# Row 1, Col 0: 50
# Row 1, Col 1: 60
# Row 1, Col 2: 70
# Row 1, Col 3: 80
```

## Question 4
**Original question:** Given `[[1, 2, 3], [4, 5, 6], [7, 8, 9]]`, extract the second column (values 2, 5, 8)

```python
table = [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]]

# Extract second column (index 1)
second_column = [row[1] for row in table]
print(f"Second column: {second_column}")  # Output: Second column: [2, 5, 8]

# Alternative without list comprehension
second_column = []
for row in table:
    second_column.append(row[1])
print(f"Second column: {second_column}")
```

## Question 5
**Original question:** Try the "wrong way" of creating a 2D list (`[[0]*3]*3`) and see what happens when you modify one element

```python
# WRONG way - creates references to same list
wrong_matrix = [[0] * 3] * 3
print("Initial matrix:")
for row in wrong_matrix:
    print(row)
# Output:
# [0, 0, 0]
# [0, 0, 0]
# [0, 0, 0]

# Modify one element
wrong_matrix[0][0] = 999

print("\nAfter modifying [0][0]:")
for row in wrong_matrix:
    print(row)
# Output:
# [999, 0, 0]  <- Changed!
# [999, 0, 0]  <- Also changed!
# [999, 0, 0]  <- Also changed!

# All rows changed because they're the same list!

# RIGHT way - creates separate lists
right_matrix = [[0] * 3 for i in range(3)]
right_matrix[0][0] = 999

print("\nCorrect way - after modifying [0][0]:")
for row in right_matrix:
    print(row)
# Output:
# [999, 0, 0]  <- Only this changed
# [0, 0, 0]
# [0, 0, 0]
```
