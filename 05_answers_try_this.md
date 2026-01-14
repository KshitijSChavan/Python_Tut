# Answers for Try This Section in 05 - Lists

## Question 1
Create a list of 5 redshifts and print the first and last values

```python
redshifts = [0.5, 1.2, 0.8, 1.7, 0.3]

first = redshifts[0]
last = redshifts[-1]

print(f"First: {first}")  # Output: First: 0.5
print(f"Last: {last}")  # Output: Last: 0.3
```

## Question 2
Take the list `[245.7, 189.3, 312.5, 198.7]` and extract the middle two values using slicing

```python
flux_list = [245.7, 189.3, 312.5, 198.7]

middle_two = flux_list[1:3]  # Index 1 and 2 (not including 3)
print(middle_two)  # Output: [189.3, 312.5]
```

## Question 3
Create an empty list, add three source names using `append()`, then remove the second one

```python
sources = []

sources.append("J1225+4011")
sources.append("J1445+3131")
sources.append("J0958+3224")
print(sources)  # Output: ['J1225+4011', 'J1445+3131', 'J0958+3224']

# Remove second one (index 1)
sources.pop(1)
print(sources)  # Output: ['J1225+4011', 'J0958+3224']

# Alternative: remove by value
# sources.remove("J1445+3131")
```

## Question 4
Use a list comprehension to create a list of squares for numbers 1 to 10

```python
squares = [i ** 2 for i in range(1, 11)]
print(squares)  # Output: [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]

# Note: range(1, 11) gives 1,2,3,...,10
```

## Question 5
Sort the list `[0.8, 1.2, 0.5, 1.7, 0.3]` in descending order

```python
redshifts = [0.8, 1.2, 0.5, 1.7, 0.3]

redshifts.sort(reverse=True)
print(redshifts)  # Output: [1.7, 1.2, 0.8, 0.5, 0.3]

# Alternative: create sorted copy without modifying original
# sorted_z = sorted(redshifts, reverse=True)
```
