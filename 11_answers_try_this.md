# Answers for Try This Section in 11 - File I/O and CSV

## Question 1
**Original question:** Create a text file with three source names, read it back and print each name

```python
# Create the file
source_names = ["J1225+4011", "J1445+3131", "J0958+3224"]

with open("source_list.txt", "w") as file:
    for name in source_names:
        file.write(name + "\n")

print("File created successfully")

# Read it back
with open("source_list.txt", "r") as file:
    for line in file:
        print(line.strip())

# Output:
# J1225+4011
# J1445+3131
# J0958+3224

# Alternative: read all lines at once
with open("source_list.txt", "r") as file:
    lines = file.readlines()
    
for line in lines:
    print(f"Source: {line.strip()}")
```

## Question 2
**Original question:** Write a list of flux values to a CSV file with a header "flux_mJy"

```python
import csv

# Flux values
flux_values = [245.7, 189.3, 312.5, 198.7, 267.4]

# Write to CSV
with open("flux_data.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["flux_mJy"])  # Header
    
    for flux in flux_values:
        writer.writerow([flux])

print("CSV file created")

# Read it back to verify
with open("flux_data.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)

# Output:
# ['flux_mJy']
# ['245.7']
# ['189.3']
# ['312.5']
# ['198.7']
# ['267.4']
```

## Question 3
**Original question:** Read a CSV file and calculate the mean of the flux column

```python
import csv

# First create a sample CSV file
data = [
    ["name", "flux"],
    ["J1225+4011", "245.7"],
    ["J1445+3131", "312.5"],
    ["J0958+3224", "198.7"],
]

with open("sources.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(data)

# Now read and calculate mean
with open("sources.csv", "r") as file:
    reader = csv.DictReader(file)
    
    flux_values = []
    for row in reader:
        flux = float(row['flux'])
        flux_values.append(flux)

mean_flux = sum(flux_values) / len(flux_values)
print(f"Mean flux: {mean_flux:.2f} mJy")
# Output: Mean flux: 252.30 mJy

# Alternative: using regular reader
with open("sources.csv", "r") as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    
    total = 0
    count = 0
    for row in reader:
        total += float(row[1])  # Flux is column 1
        count += 1

mean = total / count
print(f"Mean flux (alternative): {mean:.2f} mJy")
```

## Question 4
**Original question:** Filter a CSV to keep only rows where redshift > 1.0 and save to new file

```python
import csv

# Create sample data
sample_data = [
    ["name", "ra", "dec", "flux", "redshift"],
    ["J1225+4011", "187.7", "12.3", "245.7", "1.42"],
    ["J1445+3131", "221.2", "31.5", "312.5", "0.87"],
    ["J0958+3224", "149.5", "32.4", "198.7", "1.15"],
    ["J1543+1528", "235.8", "15.5", "289.3", "0.55"],
]

with open("all_sources.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(sample_data)

# Filter and save
with open("all_sources.csv", "r") as infile:
    reader = csv.DictReader(infile)
    
    high_z_sources = []
    for row in reader:
        if float(row['redshift']) > 1.0:
            high_z_sources.append(row)

print(f"Found {len(high_z_sources)} sources with z > 1.0")

# Write filtered data
with open("high_redshift_sources.csv", "w", newline="") as outfile:
    if high_z_sources:
        fieldnames = high_z_sources[0].keys()
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(high_z_sources)

print("Filtered catalog saved to high_redshift_sources.csv")

# Verify
with open("high_redshift_sources.csv", "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        print(f"{row['name']}: z={row['redshift']}")
# Output:
# J1225+4011: z=1.42
# J0958+3224: z=1.15
```

## Question 5
**Original question:** Handle the FileNotFoundError gracefully - print a message instead of crashing

```python
import os

filename = "nonexistent_file.csv"

# Bad way - crashes
# with open(filename, "r") as file:
#     data = file.read()  # FileNotFoundError!

# Good way - handle gracefully
try:
    with open(filename, "r") as file:
        data = file.read()
        print(f"Successfully read {filename}")
except FileNotFoundError:
    print(f"Error: File '{filename}' not found")
    print("Please check the filename and path")

# Alternative: check before opening
if os.path.exists(filename):
    with open(filename, "r") as file:
        data = file.read()
        print("File read successfully")
else:
    print(f"File '{filename}' does not exist")
    print(f"Current directory: {os.getcwd()}")
    print(f"Files in current directory: {os.listdir('.')}")

# Output:
# Error: File 'nonexistent_file.csv' not found
# Please check the filename and path
```
