# Functions

You've been using functions like `print()`, `len()`, and `range()`. Now let's write your own. Functions are reusable pieces of code - write once, use many times.

## Why Functions?

Imagine calculating the mean flux 50 times in your script:

```python
# Without functions - tedious and error-prone
total1 = sum(flux_list_1)
mean1 = total1 / len(flux_list_1)

total2 = sum(flux_list_2)
mean2 = total2 / len(flux_list_2)

# ... repeat 48 more times
```

With a function:

```python
def calculate_mean(values):
    total = sum(values)
    return total / len(values)

mean1 = calculate_mean(flux_list_1)
mean2 = calculate_mean(flux_list_2)
# ... just call the function each time
```

Functions make code shorter, clearer, and easier to fix. Change the function once and all uses get updated.

## Creating a Function

Use `def` (short for "define"):

```python
def greet():
    print("Hello from radio astronomy!")
    
greet()  # Output: Hello from radio astronomy!
greet()  # Can call it multiple times
```

The function doesn't do anything until you call it with `greet()`.

## Functions with Parameters

Most functions need input to work with:

```python
def calculate_flux_at_frequency(flux_144, freq):
    """Calculate flux at any frequency assuming spectral index -0.7"""
    spectral_index = -0.7
    flux = flux_144 * (freq / 144.0) ** spectral_index
    return flux

# Use it
flux_at_1400 = calculate_flux_at_frequency(245.7, 1400)
print(f"Flux at 1400 MHz: {flux_at_1400:.2f} mJy")
# Output: Flux at 1400 MHz: 35.57 mJy
```

The function takes `flux_144` and `freq` as inputs (parameters), does a calculation, and gives back a result with `return`.

## Return Values

Functions can send results back using `return`:

```python
def is_detected(flux, noise_level):
    """Check if source meets 5-sigma detection threshold"""
    threshold = 5 * noise_level
    return flux > threshold

detected = is_detected(145.5, 10.0)
print(detected)  # Output: True
```

Once `return` runs, the function exits immediately:

```python
def classify_flux(flux):
    if flux > 300:
        return "Bright"
    if flux > 100:
        return "Medium"
    return "Faint"

print(classify_flux(250))  # Output: Medium
```

If a function doesn't have `return`, it returns `None`:

```python
def print_info(name):
    print(f"Source: {name}")

result = print_info("J1225+4011")
print(result)  # Output: None
```

## Multiple Parameters

Functions can take several parameters:

```python
def spectral_index(flux1, freq1, flux2, freq2):
    """Calculate spectral index from two flux measurements"""
    import math
    return math.log(flux1 / flux2) / math.log(freq1 / freq2)

alpha = spectral_index(245.7, 144, 18.3, 1400)
print(f"Spectral index: {alpha:.3f}")  # Output: Spectral index: -1.112
```

The order matters when calling - match the parameter order.

## Default Parameters

Sometimes you want parameters to have default values:

```python
def calculate_luminosity(flux, distance, frequency=144):
    """Calculate luminosity. Frequency defaults to 144 MHz."""
    # Simplified calculation
    return 4 * 3.14159 * (distance ** 2) * flux

# Can call with or without frequency
lum1 = calculate_luminosity(245.7, 1000)  # Uses default freq=144
lum2 = calculate_luminosity(245.7, 1000, 323)  # Overrides default
```

Default parameters must come after non-default ones:

```python
# WRONG
# def bad_function(a=1, b):  # SyntaxError!

# CORRECT
def good_function(b, a=1):
    return a + b
```

## Docstrings

The string right after `def` is a **docstring** - it documents what the function does:

```python
def calculate_mean(values):
    """
    Calculate the arithmetic mean of a list of numbers.
    
    Parameters:
        values: list of numbers
        
    Returns:
        float: the mean value
    """
    return sum(values) / len(values)
```

Good docstrings explain what the function does, what parameters it takes, and what it returns. Use triple quotes for multi-line docstrings.

## Variable Scope

Variables created inside a function only exist inside that function:

```python
def calculate_something():
    result = 42  # Local variable
    return result

value = calculate_something()
# print(result)  # NameError! result doesn't exist outside function
```

But functions can access variables defined outside:

```python
pi = 3.14159

def circle_area(radius):
    return pi * radius ** 2  # Can use pi from outside

area = circle_area(10)
```

This is called **scope**. Generally, create variables where you need them and pass them as parameters if other functions need them.

## Lambda Functions (Brief)

For very simple functions, Python has a shorthand:

```python
# Regular function
def square(x):
    return x ** 2

# Lambda (anonymous function)
square = lambda x: x ** 2

print(square(5))  # Output: 25
```

Lambdas are useful for small operations, especially with functions like `sorted()`:

```python
sources = [("J1225", 245.7), ("J1445", 312.5), ("J0958", 198.7)]

# Sort by flux (second element of each tuple)
sorted_sources = sorted(sources, key=lambda s: s[1])
print(sorted_sources)
# Output: [('J0958', 198.7), ('J1225', 245.7), ('J1445', 312.5)]
```

But for anything complex, use regular functions - they're clearer.

## Things Worth Noting

**Functions must be defined before use:**

```python
# This fails
# result = calculate_mean([1, 2, 3])  # NameError!
# def calculate_mean(values):
#     return sum(values) / len(values)

# This works
def calculate_mean(values):
    return sum(values) / len(values)
result = calculate_mean([1, 2, 3])
```

**Mutable default arguments can surprise you:**

```python
# Avoid this pattern!
def add_source(source, catalog=[]):
    catalog.append(source)
    return catalog

cat1 = add_source("J1225")  # ['J1225']
cat2 = add_source("J1445")  # ['J1225', 'J1445'] - unexpected!
```

The default list is shared between calls. Instead:

```python
def add_source(source, catalog=None):
    if catalog is None:
        catalog = []
    catalog.append(source)
    return catalog
```

**Return vs print:**

```python
def bad_function(x):
    print(x * 2)  # Prints but doesn't return

def good_function(x):
    return x * 2  # Returns the value

result1 = bad_function(5)  # Prints 10, but result1 is None
result2 = good_function(5)  # result2 is 10
```

Use `return` to send values back. Use `print()` only for debugging or displaying results to users.

## Try This

1. Write a function that converts flux from mJy to Jy (divide by 1000)
2. Write a function that takes RA and Dec and returns them as a tuple
3. Create a function that checks if a redshift is in range 0.5-2.0 (return True/False)
4. Write a function with a default parameter for the detection threshold (default=100)
5. Add a docstring to one of your functions explaining what it does

## How This Is Typically Used in Astronomy

Functions encapsulate calculations (flux conversions, distance calculations), create reusable filters (quality checks, sample selection), organize code into logical units (one function per task), and make code testable (test each function separately).

Every analysis script uses functions to avoid repetition and keep code organized.

## Related Lessons

**Previous**: [09_for_loops.md](09_for_loops.md) - Repeating code

**Next**: [11_file_io_and_csv.md](11_file_io_and_csv.md) - Reading and writing files

**You'll use this in**: [13_statistics_as_functions.md](13_statistics_as_functions.md) - Wrapping calculations in functions
