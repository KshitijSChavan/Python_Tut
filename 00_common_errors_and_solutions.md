# Common Errors and Solutions

Everyone encounters errors when coding. This guide covers the most common ones you'll see and how to fix them. When you get an error, search this file for the error name or message.

## Installation and Import Errors

### ModuleNotFoundError: No module named 'numpy'

**What it means:** Python can't find the NumPy package.

**Solution:** Install the package:

```bash
pip3 install numpy --break-system-packages
# or
pip install numpy --break-system-packages
```

The `--break-system-packages` flag is often needed on newer Linux systems. If you still get errors, try:

```bash
python3 -m pip install numpy --break-system-packages
```

This works for any package: replace `numpy` with `matplotlib`, `scipy`, etc.

### ImportError: cannot import name 'X' from 'Y'

**What it means:** You're trying to import something that doesn't exist in that module, or you have version incompatibility.

**Common causes:**
- Typo in the import name
- Old version of the package
- Named your own file the same as a package (don't name a file `numpy.py`!)

**Solution:** Check spelling, update the package, rename your file if it conflicts:

```bash
pip3 install --upgrade numpy --break-system-packages
```

### pip: command not found

**What it means:** pip isn't installed or isn't in your PATH.

**Solution:** Use `pip3` instead, or install pip:

```bash
python3 -m ensurepip --default-pip
```

## Syntax Errors

### SyntaxError: invalid syntax

**What it means:** Python can't parse your code - there's a grammatical error.

**Common causes:**
- Missing colon after `if`, `for`, `def`, `while`
- Mismatched parentheses, brackets, or quotes
- Using `=` instead of `==` in conditions

**Examples:**

```python
# Wrong
if flux > 100
    print("Bright")

# Right
if flux > 100:
    print("Bright")

# Wrong
if flux = 100:  # Assignment, not comparison

# Right
if flux == 100:  # Comparison
```

**How to find it:** Python usually points to the line with a `^` marker. The actual error might be the line before (like a missing closing parenthesis).

### IndentationError: expected an indented block

**What it means:** Python expected indented code (after `if`, `for`, `def`, etc.) but found none.

**Solution:** Indent the code block with 4 spaces or one Tab:

```python
# Wrong
if flux > 100:
print("Bright")  # Not indented!

# Right
if flux > 100:
    print("Bright")  # Indented
```

### IndentationError: unexpected indent

**What it means:** You indented something that shouldn't be indented.

**Solution:** Remove the extra indentation or check if you accidentally mixed tabs and spaces.

### TabError: inconsistent use of tabs and spaces

**What it means:** You mixed tabs and spaces for indentation.

**Solution:** Use only spaces (4 per indent level) or only tabs, never both. Most editors can convert tabs to spaces automatically.

## Name and Type Errors

### NameError: name 'X' is not defined

**What it means:** You're using a variable or function that doesn't exist yet.

**Common causes:**
- Typo in the name
- Forgot to define the variable
- Forgot to import a module

**Examples:**

```python
# Wrong
print(fulx)  # Typo - should be 'flux'

# Wrong
print(flux)  # flux was never created

# Right
flux = 245.7
print(flux)

# Wrong
plt.plot(x, y)  # Forgot to import

# Right
import matplotlib.pyplot as plt
plt.plot(x, y)
```

### TypeError: unsupported operand type(s) for X

**What it means:** You're trying to do math with incompatible types.

**Examples:**

```python
# Wrong
result = "123" + 456  # Can't add string to int

# Right
result = int("123") + 456  # Convert string to int first
result = "123" + str(456)  # Or convert int to string

# Wrong
flux = [245.7, 189.3]
scaled = flux * 2  # Lists don't multiply like this

# Right
import numpy as np
flux = np.array([245.7, 189.3])
scaled = flux * 2  # NumPy arrays do
```

### TypeError: 'X' object is not callable

**What it means:** You're trying to call something that isn't a function.

**Common cause:** Accidentally used the same name for a variable and function:

```python
# Wrong
sum = 10 + 20  # 'sum' is now a number, not the built-in function
total = sum([1, 2, 3])  # Error! sum is no longer the function

# Right - don't overwrite built-in names
my_sum = 10 + 20
total = sum([1, 2, 3])
```

## Index and Key Errors

### IndexError: list index out of range

**What it means:** You're trying to access a list element that doesn't exist.

**Examples:**

```python
flux = [245.7, 189.3, 312.5]

# Wrong
print(flux[3])  # Only indices 0, 1, 2 exist

# Right
print(flux[2])  # Last element
print(flux[-1])  # Also gets last element

# Safe way - check length first
if len(flux) > 3:
    print(flux[3])
```

### KeyError: 'X'

**What it means:** You're trying to access a dictionary key that doesn't exist.

**Examples:**

```python
source = {"name": "J1225", "flux": 245.7}

# Wrong
print(source["redshift"])  # Key doesn't exist

# Right - check first
if "redshift" in source:
    print(source["redshift"])

# Or use get() with default
redshift = source.get("redshift", None)  # Returns None if missing
```

## Value Errors

### ValueError: invalid literal for int() with base 10

**What it means:** You're trying to convert a string to a number, but the string isn't a valid number.

**Examples:**

```python
# Wrong
flux = int("245.7")  # Can't convert float string to int directly

# Right
flux = float("245.7")  # Use float() for decimals
flux = int(float("245.7"))  # Or convert to float first, then int

# Wrong
value = int("hello")  # Not a number at all

# Right - check or handle errors
try:
    value = int(user_input)
except ValueError:
    print("That's not a valid number")
```

### ValueError: operands could not be broadcast together

**What it means:** You're doing NumPy operations on arrays with incompatible shapes.

**Examples:**

```python
import numpy as np

# Wrong
a = np.array([1, 2, 3])
b = np.array([1, 2, 3, 4])  # Different lengths
result = a + b  # Error!

# Right - make sure arrays are same shape
a = np.array([1, 2, 3, 4])
b = np.array([1, 2, 3, 4])
result = a + b
```

## Zero Division Error

### ZeroDivisionError: division by zero

**What it means:** You divided by zero.

**Solution:** Check for zero before dividing:

```python
# Wrong
result = flux / noise  # If noise is 0, error!

# Right
if noise != 0:
    result = flux / noise
else:
    result = None  # Or handle specially

# Or use NumPy which handles it gracefully
import numpy as np
result = np.divide(flux, noise, out=np.zeros_like(flux), where=noise!=0)
```

## File Errors

### FileNotFoundError: No such file or directory

**What it means:** Python can't find the file you're trying to open.

**Solutions:**
- Check the filename spelling
- Check the file is in the right directory
- Use full path: `/home/user/data/file.txt` instead of just `file.txt`
- Check current directory with `import os; print(os.getcwd())`

```python
import os

# Check if file exists before opening
if os.path.exists("data.csv"):
    with open("data.csv") as f:
        data = f.read()
else:
    print("File not found!")
```

### PermissionError: Permission denied

**What it means:** You don't have permission to read or write that file.

**Solution:** Check file permissions, or use a different location like your home directory.

## Attribute Errors

### AttributeError: 'X' object has no attribute 'Y'

**What it means:** You're trying to access a method or property that doesn't exist on that object.

**Common causes:**
- Typo in method name
- Using the wrong method for that type
- Variable is None

**Examples:**

```python
# Wrong
result = flux.append(100)  # Lists have append, but flux might not be a list

# Check type
print(type(flux))  # What is it actually?

# Wrong
data = None
mean = data.mean()  # data is None, has no mean() method

# Right - check for None first
if data is not None:
    mean = data.mean()
```

## NumPy-Specific Errors

### RuntimeWarning: invalid value encountered in divide

**What it means:** You divided by zero or got NaN in calculations. NumPy warns but continues.

**Solution:** This is often okay - NumPy uses NaN for undefined results. Clean them afterward:

```python
import numpy as np

result = np.array([1, 2, 3]) / np.array([1, 0, 3])  # Warning for middle element
clean_result = result[~np.isnan(result)]  # Remove NaN values
```

### RuntimeWarning: overflow encountered

**What it means:** Numbers got too large to represent.

**Solution:** Use log space for large numbers, or check your calculation:

```python
# Instead of
huge = 10**1000  # Overflow!

# Use
log_huge = 1000 * np.log(10)  # Work in log space
```

## Matplotlib Errors

### No module named 'tkinter'

**What it means:** Matplotlib needs tkinter for displaying plots, but it's not installed.

**Solution (Linux):**

```bash
sudo apt-get install python3-tk
```

**Or use a different backend:**

```python
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
```

### Figure is empty / No plot shows

**What it means:** You forgot `plt.show()` or saved before plotting.

**Solution:**

```python
import matplotlib.pyplot as plt

plt.plot(x, y)
plt.show()  # Don't forget this!
```

## Tips for Debugging Any Error

1. **Read the entire error message:** The last line tells you what's wrong, earlier lines tell you where.

2. **Check the line number:** Python tells you which line caused the error (though the actual problem might be earlier).

3. **Print variable values:** Add `print()` statements before the error to see what's actually in your variables.

4. **Check variable types:** Use `print(type(variable))` to see what type something actually is.

5. **Test in small pieces:** If a complex calculation fails, break it into steps and test each one.

6. **Read error messages carefully:** They often tell you exactly what's wrong.

7. **Search the error:** Copy the error message (without your specific variable names) and search online. Someone else has encountered it.

8. **Use Python's interactive mode:** Test small pieces of code in the Python interpreter to understand what's happening.

Remember: errors are normal and helpful. They tell you exactly what went wrong. Every programmer deals with them constantly, even experts.
