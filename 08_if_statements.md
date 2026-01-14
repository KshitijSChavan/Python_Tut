# If Statements and Logical Operators

So far our code runs line by line, always doing the same thing. But often you need to make decisions: "If this flux is above the threshold, include it in the catalog." "If the quality flag is bad, skip this source." That's what **if statements** do.

## Basic If Statement

An if statement runs code only when a condition is True:

```python
flux = 245.7
threshold = 100.0

if flux > threshold:
    print("Source detected!")
    print(f"Flux: {flux} mJy")
    
# Output:
# Source detected!
# Flux: 245.7 mJy
```

The condition `flux > threshold` evaluates to True or False. If True, the indented code runs. If False, Python skips it entirely.

**Indentation is crucial here.** Python uses it to know what code belongs to the if statement. Use 4 spaces or one Tab. If you forget to indent, you'll get an `IndentationError`. If you mix tabs and spaces, you'll also get errors - pick one and stick with it.

## If-Else: Two Paths

Often you want to do one thing if a condition is True, and something different if it's False:

```python
flux = 75.5
threshold = 100.0

if flux > threshold:
    print("Source detected")
else:
    print("Below detection threshold")
    
# Output: Below detection threshold
```

Think of `else` as "otherwise, do this instead."

## If-Elif-Else: Multiple Possibilities

When you have more than two cases, use `elif` (short for "else if"):

```python
spectral_index = -0.75

if spectral_index < -0.5:
    classification = "Steep spectrum"
elif spectral_index > 0.5:
    classification = "Inverted spectrum"
else:
    classification = "Flat spectrum"
    
print(classification)  # Output: Steep spectrum
```

Here's the important part: Python checks each condition in order and runs the first one that's True, then stops. Even if later conditions would also be True, they're ignored. This means order matters:

```python
value = 150

if value > 100:
    print("Greater than 100")  # This runs
elif value > 50:
    print("Greater than 50")  # This is also True but never checked!
```

So arrange your conditions from most specific to most general.

## Combining Conditions

Use `and`, `or`, and `not` to check multiple things at once:

```python
flux = 245.7
size = 45.0

# Both must be True
if flux > 200 and size > 30:
    print("Bright and extended source")

# At least one must be True  
if flux > 500 or size > 100:
    print("Either very bright or very large")

# Negate a condition
quality_flag = 0
if not quality_flag:  # Same as: if quality_flag == 0
    print("Good quality data")
```

When combining multiple conditions, use parentheses for clarity:

```python
if (flux > 200 and size > 30) or quality_flag == 0:
    print("Either (bright AND extended) OR good quality")
```

## Practical Patterns

**Checking if a value is in a list:**

```python
telescope = "LOFAR"

if telescope in ["LOFAR", "GMRT", "VLA"]:
    print("Radio telescope")
```

This is much cleaner than chaining multiple `or` statements.

**Range checking:**

```python
redshift = 1.2

if 0.5 <= redshift <= 2.0:
    print("In target redshift range")
```

Python lets you chain comparisons like you would write them mathematically.

**Handling missing data:**

```python
measurement = None  # Missing value

if measurement is not None:
    print(f"Value: {measurement}")
else:
    print("No measurement available")
```

Use `is` (not `==`) when checking for `None`.

## Things Worth Noting

**Assignment vs comparison - a common mistake:**

```python
flux = 245.7  # Single = assigns a value

if flux == 245.7:  # Double == compares values
    print("Match")
    
# if flux = 245.7:  # SyntaxError! Can't use = in a condition
```

**Empty values are "falsy":**

```python
source_list = []

if source_list:
    print("Has sources")
else:
    print("Empty list")  # This runs because [] is False-like
```

Empty lists, empty strings `""`, zero `0`, and `None` are all treated as False in conditions. Everything else is True. This can be handy for checking if a list has items.

**Nested ifs get messy quickly:**

```python
# This gets hard to read
if flux > 200:
    if spectral_index < -0.5:
        if redshift > 1.0:
            print("Interesting source!")

# Often better to combine conditions
if flux > 200 and spectral_index < -0.5 and redshift > 1.0:
    print("Interesting source!")
```

## Try This

1. Write an if-elif-else that classifies flux into "Bright" (>300), "Medium" (100-300), or "Faint" (<100)
2. Check if a source with redshift=1.5 and flux=250 meets criteria: 1.0 < z < 2.0 AND flux > 200
3. Check if a telescope name is in the list ["LOFAR", "GMRT", "VLA"]
4. Write code that skips processing if quality_flag is not 0

## How This Is Typically Used in Astronomy

Every data pipeline uses if statements to filter and classify sources: skip bad quality flags, include only sources meeting flux and redshift criteria, classify sources as steep/flat spectrum or compact/extended, and handle missing or invalid measurements.

## Related Lessons

**Previous**: [07_nested_lists.md](07_nested_lists.md) - 2D data structures

**Next**: [09_for_loops.md](09_for_loops.md) - Repeating code

**Uses**: [04_booleans_and_comparisons.md](04_booleans_and_comparisons.md) - The conditions we test
