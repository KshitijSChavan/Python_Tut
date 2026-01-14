# Answers for Try This Section in 01 - Variables and Types

## Question 1
Create variables for RA (187.5), Dec (12.3), flux (245.7), and source name

```python
ra = 187.5
dec = 12.3
flux = 245.7
source_name = "J1225+4011"

print(ra, dec, flux, source_name)
# Output: 187.5 12.3 245.7 J1225+4011
```

## Question 2
Convert "1.42" to a float and multiply by 2

```python
z_string = "1.42"
z = float(z_string)
result = z * 2

print(result)  # Output: 2.84
```

## Question 3
Swap two variables without a temporary variable

```python
a = 10
b = 20
print(f"Before: a={a}, b={b}")  # Output: Before: a=10, b=20

a, b = b, a
print(f"After: a={a}, b={b}")  # Output: After: a=20, b=10
```
