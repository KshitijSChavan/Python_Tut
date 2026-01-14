# File I/O and CSV

Real data doesn't live in your Python scripts - it's in files. You need to read catalog files, observation logs, and data tables. This is how you get data into Python and save results back out.

## Reading a Simple Text File

The pattern you'll use most often is `with open()`. This opens a file, lets you work with it, then automatically closes it when done - even if an error occurs:

```python
with open("observation_log.txt", "r") as file:
    content = file.read()
    print(content)
```

The `"r"` means read mode. The `with` statement handles cleanup - you don't need to remember to close the file.

What happens if the file doesn't exist? You get a `FileNotFoundError`. If you see this, double-check the filename and make sure the file is in the directory where you're running Python.

## Reading Line by Line

For large files, reading everything at once wastes memory. Read line by line instead:

```python
with open("source_list.txt", "r") as file:
    for line in file:
        print(line.strip())  # strip() removes the \n newline character
```

Each line includes a newline character at the end, which is why we use `strip()`. Without it, you'd get extra blank lines in your output.

You can also load all lines into a list if you need to process them multiple times:

```python
with open("source_list.txt", "r") as file:
    lines = file.readlines()  # Returns a list of strings
    
for line in lines:
    print(line.strip())
```

## Writing Text Files

Use `"w"` mode to write. Important warning: this **overwrites the file** if it already exists:

```python
sources = ["J1225+4011", "J1445+3131", "J0958+3224"]

with open("output.txt", "w") as file:
    for source in sources:
        file.write(source + "\n")  # Must add newline yourself
```

Unlike `print()`, `write()` doesn't add newlines automatically. You have to include `\n` explicitly.

If you want to add to an existing file without erasing it, use `"a"` (append mode):

```python
with open("output.txt", "a") as file:
    file.write("J1543+1528\n")  # Adds to end
```

## Understanding CSV Files

CSV (comma-separated values) files are how spreadsheets and databases export data. Each line is a row, commas separate columns:

```
name,ra,dec,flux
J1225+4011,187.7,12.3,245.7
J1445+3131,221.2,31.5,312.5
J0958+3224,149.5,32.4,198.7
```

You can parse this manually by splitting on commas:

```python
with open("sources.csv", "r") as file:
    lines = file.readlines()
    
# First line is the header
header = lines[0].strip().split(",")
print(header)  # ['name', 'ra', 'dec', 'flux']

# Process each data line
for line in lines[1:]:  # Skip header with [1:]
    parts = line.strip().split(",")
    name = parts[0]
    ra = float(parts[1])
    dec = float(parts[2])
    flux = float(parts[3])
    print(f"{name}: {flux} mJy at ({ra}, {dec})")
```

The `split(",")` method breaks the line wherever it finds a comma, giving you a list. Then you convert the numeric strings to actual numbers with `float()`.

## Using Python's CSV Module

The manual approach works but has problems - what if a field contains a comma? What about quotes? Python's `csv` module handles these edge cases:

```python
import csv

with open("sources.csv", "r") as file:
    reader = csv.reader(file)
    header = next(reader)  # Get first line
    
    for row in reader:
        name, ra, dec, flux = row
        print(f"{name}: {flux} mJy")
```

Even better, use `DictReader` to access columns by name instead of position:

```python
import csv

with open("sources.csv", "r") as file:
    reader = csv.DictReader(file)  # Automatically uses first row as header
    
    for row in reader:
        print(f"{row['name']}: {row['flux']} mJy")
```

This is more readable and less error-prone. If someone adds a column, your code doesn't break because you're referencing by name, not position.

## Writing CSV Files

```python
import csv

sources = [
    ["J1225+4011", 187.7, 12.3, 245.7],
    ["J1445+3131", 221.2, 31.5, 312.5],
    ["J0958+3224", 149.5, 32.4, 198.7]
]

with open("output.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["name", "ra", "dec", "flux"])  # Write header
    writer.writerows(sources)  # Write all data rows at once
```

The `newline=""` parameter prevents extra blank lines on Windows - always include it when writing CSV files.

## File Paths Matter

If your file is in the same directory where you're running Python, just use the filename. But if it's elsewhere, you need to specify the path:

```python
# In a subdirectory called 'data'
file_path = "data/sources.csv"

# In the parent directory
file_path = "../sources.csv"

# Absolute path (full location on your system)
file_path = "/home/user/astronomy/data/sources.csv"
```

On Windows, paths look different but Python handles them if you use forward slashes or raw strings:

```python
# Both work on Windows
file_path = "C:/data/sources.csv"
file_path = r"C:\data\sources.csv"  # r makes it a raw string
```

Before trying to read a file, you can check if it exists:

```python
import os

if os.path.exists("sources.csv"):
    # File exists, safe to read
    with open("sources.csv", "r") as file:
        content = file.read()
else:
    print("File not found - check the path")
```

This prevents crashes and gives helpful messages.

## Practical Example: Filtering a Catalog

Here's a common workflow - read a catalog, apply quality cuts, write the filtered results:

```python
import csv

# Read full catalog and filter bright sources
bright_sources = []

with open("full_catalog.csv", "r") as infile:
    reader = csv.DictReader(infile)
    
    for row in reader:
        flux = float(row['flux'])
        if flux > 200:  # Quality cut
            bright_sources.append(row)

print(f"Found {len(bright_sources)} bright sources out of full catalog")

# Write filtered catalog to new file
with open("bright_sources.csv", "w", newline="") as outfile:
    if bright_sources:  # Check list isn't empty
        fieldnames = bright_sources[0].keys()  # Get column names
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(bright_sources)
```

This pattern - read, filter, write - is fundamental to data processing.

## Things Worth Noting

**Always use `with open()` instead of manually opening and closing.** If an error occurs between `open()` and `close()`, the file stays open and can cause problems. The `with` statement guarantees cleanup.

**File modes to know:**
- `"r"` - read (file must exist or you get FileNotFoundError)
- `"w"` - write (creates new file or erases existing one)
- `"a"` - append (adds to end of existing file)

**Text vs binary mode:** For text files and CSV, use `"r"` or `"w"`. For images or FITS files (which we'll cover later), you need binary mode: `"rb"` or `"wb"`.

**Encoding issues:** If you see weird characters like mojibake (�), your file might use a different encoding. Specify it explicitly:

```python
with open("file.txt", "r", encoding="utf-8") as file:
    content = file.read()
```

UTF-8 is the most common encoding. If that doesn't work, try `"latin-1"` or `"iso-8859-1"`.

**CSV with different separators:** Not all "CSV" files use commas. Some use tabs (TSV files) or semicolons. Specify the delimiter:

```python
reader = csv.reader(file, delimiter="\t")  # For tabs
reader = csv.reader(file, delimiter=";")   # For semicolons
```

## Try This

1. Create a text file with three source names, read it back and print each name
2. Write a list of flux values to a CSV file with a header "flux_mJy"
3. Read a CSV file and calculate the mean of the flux column
4. Filter a CSV to keep only rows where redshift > 1.0 and save to new file
5. Handle the FileNotFoundError gracefully - print a message instead of crashing

## How This Is Typically Used in Astronomy

Reading observational data from CSV tables, loading source catalogs from text files, writing analysis results for later use, processing multiple files in batch, and converting between file formats.

Later we'll use NumPy to load data more efficiently, but CSV and text files remain important for smaller datasets and compatibility.

## Related Lessons

**Previous**: [10_functions.md](10_functions.md) - Reusable code

**Next**: [12_statistics_from_scratch.md](12_statistics_from_scratch.md) - Calculating statistical measures

**Better for large datasets**: [15_numpy_data_loading.md](15_numpy_data_loading.md) - NumPy's faster file reading
