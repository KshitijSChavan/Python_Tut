# Python for Radio Astronomy Research

A comprehensive Python learning path designed for radio astronomers transitioning from research to programming. This curriculum covers fundamentals through advanced scientific computing with astronomy-specific examples throughout.

## Repository Structure

### Main Lessons (23 files)
- **00_common_errors_and_solutions.md** - Comprehensive debugging guide
- **01-10** - Python fundamentals (variables, arithmetic, strings, booleans, lists, data structures, conditionals, loops, functions)
- **11-13** - File I/O and statistics (reading/writing files, manual statistics implementations, reusable functions)
- **14-15** - NumPy (arrays, vectorization, data loading, boolean indexing)
- **16-17** - Matplotlib (basic and advanced plotting, subplots, customization)
- **18** - Astronomy-specific calculations (unit conversions, spectral indices, coordinate systems)
- **19** - SciPy essentials (curve fitting, statistical tests, interpolation, integration)
- **20** - Working with real data (handling missing values, outliers, cross-matching catalogs)
- **21** - Best practices (code organization, documentation, error handling, debugging)
- **22** - Advanced topics (Monte Carlo simulations, correlation analysis, power laws, optimization)

### Answer Files (22 files)
Each lesson includes a corresponding `XX_answers_try_this.md` file with:
- Original "Try This" question stated clearly
- Complete working code solutions
- Expected output as comments
- Alternative approaches where relevant
- Explanatory notes

## Design Philosophy

### Text-First Approach
- Explanations precede code blocks
- Context provided for why each feature exists
- Practical use cases described before implementation
- Natural conversational flow throughout

### Astronomy Context
Every lesson includes domain-specific examples:
- Radio source catalogs (LoTSS, FIRST, NVSS)
- Flux measurements and spectral analysis
- Coordinate systems and position matching
- Multi-wavelength observations
- Time-series analysis

### Error Awareness
Common pitfalls integrated naturally:
- FileNotFoundError handling in I/O lessons
- ZeroDivisionError warnings in calculations
- NaN handling in data processing
- IndexError prevention in array operations

### Progressive Complexity
- Start with single values, progress to lists, then NumPy arrays
- Manual implementations before using libraries
- Simple examples before complex workflows
- Always explain the "why" before the "how"

## Learning Path

### Beginners (Lessons 01-10)
Start here if you're new to programming. These lessons cover:
- Basic data types and operations
- Control flow (if statements, loops)
- Functions and code organization
- No prerequisites required

### Intermediate (Lessons 11-18)
For those comfortable with Python basics:
- Working with files and real data
- Scientific computing with NumPy
- Data visualization with Matplotlib
- Domain-specific calculations

### Advanced (Lessons 19-22)
For experienced programmers wanting scientific tools:
- SciPy for advanced analysis
- Production-quality data pipelines
- Code organization and best practices
- Monte Carlo methods and optimization

## Key Features

### No Mechanical Patterns
Lessons avoid the repetitive "subtitle → code block → subtitle → code block" pattern. Instead:
- Code illustrates concepts naturally
- Explanations flow conversationally
- Examples build on each other organically

### Practical Examples
All code examples use realistic scenarios:
- Analyzing radio galaxy catalogs
- Calculating spectral indices
- Fitting power laws to source counts
- Cross-matching multi-wavelength observations
- Handling time-series data with gaps

### Check-Before-Install
Package lessons start by checking if already installed:
```bash
python3 -c "import numpy; print(numpy.__version__)"
```
Then show installation only if needed.

## Usage Guide

### Sequential Learning
Work through lessons in order. Each builds on previous concepts.

### Try This Sections
Every lesson ends with 5 hands-on exercises. Solutions are in corresponding answer files.

### Error Reference
When you encounter an error, check `00_common_errors_and_solutions.md` first. It covers:
- Installation and import errors
- Syntax errors (indentation, colons, quotes)
- Type and value errors
- Index and key errors
- File and permission errors
- NumPy-specific warnings

## Prerequisites

- Basic Linux/terminal familiarity
- Python 3.x installed
- Text editor or IDE (VS Code, PyCharm, vim, etc.)
- Astronomy research background (helpful but not required)

## Installation

Packages covered in lessons:
```bash
pip3 install numpy matplotlib scipy --break-system-packages
```

## Target Audience

This curriculum was designed for:
- Radio astronomers transitioning from research to programming
- Researchers familiar with data analysis but new to Python
- Anyone needing practical scientific computing skills
- Students wanting astronomy-context examples

## What Makes This Different

### From Typical Tutorials
- Every example uses real astronomy scenarios
- Error handling integrated from the start
- Explains *why* things work, not just *how*
- Professional coding practices emphasized

### From University Courses
- Focused on practical skills over theory
- Immediate application to research workflows
- Real-world data handling challenges
- Production-quality code patterns

## Contributing

This is a teaching resource. If you find errors or have suggestions:
- Check if the issue is addressed in error reference (file 00)
- Verify examples work with current package versions
- Ensure new examples maintain astronomy context

## License

Educational resource for astronomy research community.

## Acknowledgments

Designed for researchers by researchers. Examples drawn from:
- LoTSS (LOFAR Two-metre Sky Survey)
- Radio AGN studies
- Giant radio galaxy projects
- Multi-wavelength catalog work

---

**Start with:** `01_variables_and_types.md`

**Got errors?** Check: `00_common_errors_and_solutions.md`

**Need help?** Each "Try This" section has complete solutions in answer files.
