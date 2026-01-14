# SciPy - Complete Guide

## Introduction

SciPy (Scientific Python) is a collection of mathematical algorithms and convenience functions built on NumPy. It provides high-level scientific and technical computing capabilities including optimization, integration, interpolation, signal processing, linear algebra, statistics, and more.

**Key Capabilities:**
- Optimization and root finding
- Integration and differential equations
- Interpolation
- Signal processing and filtering
- Statistical functions and distributions
- Linear algebra (advanced operations)
- Sparse matrices
- Spatial data structures and algorithms

**Why SciPy?**
- Built on NumPy (fast and efficient)
- Well-tested, reliable implementations
- Comprehensive scientific algorithms
- Standard tool in scientific Python

**Installation:**
```bash
pip install scipy
```

---

## Example 1: Curve Fitting and Optimization

```python
import numpy as np
from scipy.optimize import curve_fit, minimize
import matplotlib.pyplot as plt

# Generate synthetic data with noise
np.random.seed(42)
x_data = np.linspace(0, 10, 50)
y_true = 2.5 * np.exp(-0.5 * x_data) + 1.0
y_data = y_true + np.random.normal(0, 0.2, len(x_data))

# Define model function
def exponential_model(x, a, b, c):
    """Model: y = a * exp(-b * x) + c"""
    return a * np.exp(-b * x) + c

# Fit the model to data
initial_guess = [2.0, 0.5, 1.0]
params, covariance = curve_fit(exponential_model, x_data, y_data, p0=initial_guess)
a_fit, b_fit, c_fit = params

# Calculate parameter uncertainties
param_errors = np.sqrt(np.diag(covariance))

print("=== Curve Fitting Results ===")
print(f"Parameter a: {a_fit:.4f} ± {param_errors[0]:.4f}")
print(f"Parameter b: {b_fit:.4f} ± {param_errors[1]:.4f}")
print(f"Parameter c: {c_fit:.4f} ± {param_errors[2]:.4f}")
print()

# Calculate R-squared
y_pred = exponential_model(x_data, a_fit, b_fit, c_fit)
ss_res = np.sum((y_data - y_pred) ** 2)
ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
r_squared = 1 - (ss_res / ss_tot)
print(f"R-squared: {r_squared:.4f}")
print()

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, label='Data', alpha=0.6, s=50)
plt.plot(x_data, y_true, 'g--', label='True function', linewidth=2)
x_smooth = np.linspace(0, 10, 200)
y_fit = exponential_model(x_smooth, a_fit, b_fit, c_fit)
plt.plot(x_smooth, y_fit, 'r-', label='Fitted function', linewidth=2)
plt.xlabel('X', fontsize=12)
plt.ylabel('Y', fontsize=12)
plt.title('Exponential Curve Fitting', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('curve_fitting.png', dpi=300, bbox_inches='tight')
plt.show()

# Optimization example: Find minimum of a function
def rosenbrock(x):
    """Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2"""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

# Starting point
x0 = np.array([0, 0])

# Minimize the function
result = minimize(rosenbrock, x0, method='BFGS')

print("=== Optimization Results ===")
print(f"Minimum found at: x = {result.x[0]:.6f}, y = {result.x[1]:.6f}")
print(f"Function value at minimum: {result.fun:.10f}")
print(f"Number of iterations: {result.nit}")
print(f"Success: {result.success}")
```

### Code Explanation:

**Line 7-9:** Generate synthetic data with noise to demonstrate curve fitting. The true function is an exponential decay plus a constant.

**Line 12-14:** Define the model function we want to fit. It takes x and three parameters (a, b, c) and returns the predicted y values according to the model equation.

**Line 17:** `curve_fit(func, xdata, ydata, p0)` finds optimal parameters:
- `func` - the model function
- `xdata`, `ydata` - the data to fit
- `p0` - initial guess for parameters (helps convergence)
Returns: optimal parameters and covariance matrix

**Line 21:** `covariance` is a matrix describing parameter uncertainties and correlations. Diagonal elements are variances of parameters.

**Line 22:** `np.sqrt(np.diag(covariance))` extracts standard errors (uncertainties) of each parameter from the diagonal of covariance matrix.

**Line 32-34:** Calculate R-squared (coefficient of determination):
- `ss_res` - sum of squared residuals (prediction errors)
- `ss_tot` - total sum of squares (variance in data)
- R² = 1 means perfect fit, R² = 0 means model no better than mean

**Line 56-58:** Define Rosenbrock function, a common test function for optimization. Has global minimum at (1, 1).

**Line 64:** `minimize(func, x0, method)` finds minimum of function:
- `func` - function to minimize
- `x0` - starting point for search
- `method` - optimization algorithm ('BFGS', 'Nelder-Mead', 'Powell', etc.)

**Line 67:** `result.x` contains the optimal parameters found.

**Line 68:** `result.fun` is the function value at the minimum.

**Line 69:** `result.nit` is the number of iterations taken.

**Line 70:** `result.success` indicates whether optimization converged successfully.

---

## Example 2: Integration and Differential Equations

```python
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

# Example 1: Definite integral
def gaussian(x):
    """Gaussian function"""
    return np.exp(-x**2)

# Integrate from -infinity to infinity
result, error = integrate.quad(gaussian, -np.inf, np.inf)
print("=== Numerical Integration ===")
print(f"∫ exp(-x²) dx from -∞ to ∞ = {result:.10f}")
print(f"Analytical result: √π = {np.sqrt(np.pi):.10f}")
print(f"Integration error estimate: {error:.2e}")
print()

# Example 2: Integrate discrete data (trapezoid rule)
x_discrete = np.linspace(0, 10, 100)
y_discrete = np.sin(x_discrete) * np.exp(-0.1 * x_discrete)
area = integrate.trapezoid(y_discrete, x_discrete)
print(f"Area under curve (trapezoid): {area:.6f}")
print()

# Example 3: Solve ordinary differential equation (ODE)
# Solve: dy/dt = -0.5*y, y(0) = 10
def exponential_decay(t, y):
    """ODE: dy/dt = -0.5*y"""
    return -0.5 * y

# Initial condition and time span
y0 = [10]
t_span = (0, 10)
t_eval = np.linspace(0, 10, 100)

# Solve ODE
solution = integrate.solve_ivp(exponential_decay, t_span, y0, t_eval=t_eval)

# Analytical solution for comparison
y_analytical = 10 * np.exp(-0.5 * t_eval)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(solution.t, solution.y[0], 'b-', label='Numerical solution', linewidth=2)
plt.plot(t_eval, y_analytical, 'r--', label='Analytical solution', linewidth=2)
plt.xlabel('Time', fontsize=12)
plt.ylabel('y(t)', fontsize=12)
plt.title('Solving ODE: dy/dt = -0.5y, y(0) = 10', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ode_solution.png', dpi=300, bbox_inches='tight')
plt.show()

# Example 4: System of ODEs (Lotka-Volterra predator-prey)
def lotka_volterra(t, y):
    """Predator-prey equations"""
    prey, predator = y
    alpha = 1.5    # prey birth rate
    beta = 1.0     # predation rate
    delta = 0.75   # predator efficiency
    gamma = 1.0    # predator death rate
    
    dprey_dt = alpha * prey - beta * prey * predator
    dpredator_dt = delta * prey * predator - gamma * predator
    
    return [dprey_dt, dpredator_dt]

# Initial populations
y0_system = [2.0, 1.0]  # [prey, predator]
t_span_system = (0, 20)
t_eval_system = np.linspace(0, 20, 1000)

# Solve system
solution_system = integrate.solve_ivp(lotka_volterra, t_span_system, y0_system, 
                                     t_eval=t_eval_system, method='RK45')

# Plot phase space and time series
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Time series
ax1.plot(solution_system.t, solution_system.y[0], 'b-', label='Prey', linewidth=2)
ax1.plot(solution_system.t, solution_system.y[1], 'r-', label='Predator', linewidth=2)
ax1.set_xlabel('Time', fontsize=12)
ax1.set_ylabel('Population', fontsize=12)
ax1.set_title('Lotka-Volterra: Population Dynamics', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Phase space
ax2.plot(solution_system.y[0], solution_system.y[1], 'g-', linewidth=2)
ax2.plot(solution_system.y[0][0], solution_system.y[1][0], 'go', markersize=10, 
         label='Start')
ax2.set_xlabel('Prey Population', fontsize=12)
ax2.set_ylabel('Predator Population', fontsize=12)
ax2.set_title('Phase Space', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lotka_volterra.png', dpi=300, bbox_inches='tight')
plt.show()

print("Lotka-Volterra system solved successfully")
```

### Code Explanation:

**Line 6-8:** Define a function to integrate. Must take single variable as input.

**Line 11:** `integrate.quad(func, a, b)` performs numerical integration (quadrature):
- `func` - function to integrate
- `a, b` - integration limits (can use `np.inf` for infinity)
Returns: (result, error_estimate)

Uses adaptive algorithm that automatically refines where needed for accuracy.

**Line 21:** `integrate.trapezoid(y, x)` integrates discrete data points using trapezoid rule:
- `y` - function values at points
- `x` - x-coordinates of points
Approximates area by connecting points with straight lines and summing trapezoid areas.

**Line 26-28:** Define ODE as function `f(t, y)` that returns dy/dt. SciPy format requires this specific signature.

**Line 31-33:** Set up problem:
- `y0` - initial condition(s) as list
- `t_span` - tuple (t_start, t_end) for integration
- `t_eval` - specific times where solution is needed (optional)

**Line 36:** `integrate.solve_ivp(func, t_span, y0, t_eval)` solves initial value problem:
- `func` - ODE function
- `t_span` - integration interval
- `y0` - initial values
- `t_eval` - times to evaluate solution
- `method` - solver ('RK45', 'RK23', 'DOP853', 'Radau', 'BDF')

Returns solution object with `.t` (times) and `.y` (solution values).

**Line 60-68:** Define system of ODEs. Function returns list with derivative of each variable. For predator-prey:
- Prey grows exponentially, reduced by predation
- Predators grow when eating prey, die naturally

**Line 71:** Initial conditions for system must match number of equations (2 populations).

**Line 76-77:** `solve_ivp` handles systems automatically - just provide list of initial conditions and return list of derivatives.

**Line 87:** `solution.y[0]` accesses first variable (prey), `solution.y[1]` accesses second (predator).

**Line 94:** Phase space plot shows predator vs prey population, revealing cyclic dynamics.

---

## Example 3: Interpolation and Smoothing

```python
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

# Generate sparse data with noise
np.random.seed(42)
x_sparse = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_sparse = np.sin(x_sparse) + 0.1 * np.random.randn(len(x_sparse))

# Create different interpolation functions
f_linear = interpolate.interp1d(x_sparse, y_sparse, kind='linear')
f_quadratic = interpolate.interp1d(x_sparse, y_sparse, kind='quadratic')
f_cubic = interpolate.interp1d(x_sparse, y_sparse, kind='cubic')

# Create smooth x values for plotting
x_smooth = np.linspace(0, 10, 200)

# Evaluate interpolations
y_linear = f_linear(x_smooth)
y_quadratic = f_quadratic(x_smooth)
y_cubic = f_cubic(x_smooth)

# Plot comparison
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(x_sparse, y_sparse, 'o', label='Data points', markersize=8)
plt.plot(x_smooth, y_linear, '-', label='Linear', linewidth=2)
plt.title('Linear Interpolation', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.plot(x_sparse, y_sparse, 'o', label='Data points', markersize=8)
plt.plot(x_smooth, y_quadratic, '-', label='Quadratic', linewidth=2)
plt.title('Quadratic Interpolation', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
plt.plot(x_sparse, y_sparse, 'o', label='Data points', markersize=8)
plt.plot(x_smooth, y_cubic, '-', label='Cubic', linewidth=2)
plt.title('Cubic Interpolation', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
plt.plot(x_sparse, y_sparse, 'ko', label='Data', markersize=8, zorder=3)
plt.plot(x_smooth, y_linear, '-', alpha=0.5, label='Linear', linewidth=2)
plt.plot(x_smooth, y_quadratic, '-', alpha=0.5, label='Quadratic', linewidth=2)
plt.plot(x_smooth, y_cubic, '-', alpha=0.5, label='Cubic', linewidth=2)
plt.title('All Methods Compared', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('interpolation_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 2D interpolation example
x_2d = np.linspace(0, 3, 4)
y_2d = np.linspace(0, 3, 4)
z_2d = np.array([[1, 2, 3, 4],
                 [2, 3, 4, 5],
                 [3, 4, 5, 6],
                 [4, 5, 6, 7]])

# Create 2D interpolation function
f_2d = interpolate.interp2d(x_2d, y_2d, z_2d, kind='cubic')

# Evaluate on finer grid
x_fine = np.linspace(0, 3, 20)
y_fine = np.linspace(0, 3, 20)
z_fine = f_2d(x_fine, y_fine)

# Plot 2D interpolation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Original data
im1 = ax1.imshow(z_2d, extent=[0, 3, 0, 3], origin='lower', cmap='viridis')
ax1.set_title('Original 4x4 Data', fontweight='bold')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
plt.colorbar(im1, ax=ax1)

# Interpolated data
im2 = ax2.imshow(z_fine, extent=[0, 3, 0, 3], origin='lower', cmap='viridis')
ax2.set_title('Interpolated 20x20 Data', fontweight='bold')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
plt.colorbar(im2, ax=ax2)

plt.tight_layout()
plt.savefig('2d_interpolation.png', dpi=300, bbox_inches='tight')
plt.show()

# Univariate spline (smoothing)
x_noisy = np.linspace(0, 10, 50)
y_noisy = np.sin(x_noisy) + 0.3 * np.random.randn(len(x_noisy))

# Fit spline with different smoothing factors
spline_smooth0 = interpolate.UnivariateSpline(x_noisy, y_noisy, s=0)  # No smoothing
spline_smooth1 = interpolate.UnivariateSpline(x_noisy, y_noisy, s=1)  # Some smoothing
spline_smooth5 = interpolate.UnivariateSpline(x_noisy, y_noisy, s=5)  # More smoothing

x_dense = np.linspace(0, 10, 500)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(x_noisy, y_noisy, 'o', alpha=0.5, label='Noisy data')
plt.plot(x_dense, spline_smooth0(x_dense), 'r-', label='s=0 (interpolate)', linewidth=2)
plt.plot(x_dense, np.sin(x_dense), 'g--', label='True function', linewidth=2)
plt.title('Smoothing s=0', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(x_noisy, y_noisy, 'o', alpha=0.5, label='Noisy data')
plt.plot(x_dense, spline_smooth1(x_dense), 'r-', label='s=1', linewidth=2)
plt.plot(x_dense, np.sin(x_dense), 'g--', label='True function', linewidth=2)
plt.title('Smoothing s=1', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.plot(x_noisy, y_noisy, 'o', alpha=0.5, label='Noisy data')
plt.plot(x_dense, spline_smooth5(x_dense), 'r-', label='s=5', linewidth=2)
plt.plot(x_dense, np.sin(x_dense), 'g--', label='True function', linewidth=2)
plt.title('Smoothing s=5', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('spline_smoothing.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Code Explanation:

**Line 11-13:** `interpolate.interp1d(x, y, kind)` creates interpolation function:
- `x, y` - known data points
- `kind` - interpolation type: 'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'
Returns a function that can be called with new x values.

**Line 16:** Create fine grid of x values to evaluate interpolation and produce smooth curves.

**Line 19-21:** Call interpolation functions like regular functions: `f(x_new)` returns interpolated y values.

**Line 66:** `interpolate.interp2d(x, y, z, kind)` creates 2D interpolation function:
- `x, y` - 1D coordinate arrays
- `z` - 2D array of function values
- `kind` - 'linear', 'cubic', 'quintic'
Returns function that takes new x and y arrays.

**Line 71:** Call 2D interpolation: `f_2d(x_new, y_new)` returns 2D array of interpolated values.

**Line 105-107:** `interpolate.UnivariateSpline(x, y, s)` fits smoothing spline:
- `x, y` - data points
- `s` - smoothing factor:
  - `s=0` passes through all points (no smoothing)
  - `s>0` allows deviation from points (more smoothing)
  - Larger s = smoother curve

Returns callable spline object that can compute values and derivatives.

**Interpolation Notes:**
- Linear: connects points with straight lines (C0 continuity)
- Quadratic: uses parabolas (C1 continuity)
- Cubic: uses cubic polynomials (C2 continuity - smooth derivatives)
- Don't extrapolate far beyond data range - unreliable

---

## Example 4: Statistical Functions and Distributions

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Example 1: Statistical tests
# Generate two samples
np.random.seed(42)
sample1 = np.random.normal(100, 15, 100)
sample2 = np.random.normal(105, 15, 100)

print("=== Statistical Tests ===")

# T-test: Are means significantly different?
t_stat, p_value_ttest = stats.ttest_ind(sample1, sample2)
print(f"T-test: t={t_stat:.4f}, p-value={p_value_ttest:.4f}")
if p_value_ttest < 0.05:
    print("  → Samples have significantly different means (p < 0.05)")
else:
    print("  → No significant difference in means (p ≥ 0.05)")
print()

# Kolmogorov-Smirnov test: Are distributions different?
ks_stat, p_value_ks = stats.ks_2samp(sample1, sample2)
print(f"KS test: statistic={ks_stat:.4f}, p-value={p_value_ks:.4f}")
if p_value_ks < 0.05:
    print("  → Samples from different distributions (p < 0.05)")
else:
    print("  → Samples from same distribution (p ≥ 0.05)")
print()

# Chi-square test for goodness of fit
observed = np.array([15, 25, 35, 25])
expected = np.array([20, 25, 30, 25])
chi2_stat, p_value_chi2 = stats.chisquare(observed, expected)
print(f"Chi-square test: χ²={chi2_stat:.4f}, p-value={p_value_chi2:.4f}")
print()

# Example 2: Probability distributions
# Normal distribution
print("=== Normal Distribution ===")
mu, sigma = 100, 15
norm_dist = stats.norm(loc=mu, scale=sigma)

print(f"Mean: {norm_dist.mean()}")
print(f"Std: {norm_dist.std()}")
print(f"PDF at x=100: {norm_dist.pdf(100):.6f}")
print(f"CDF at x=115: {norm_dist.cdf(115):.4f}")  # P(X ≤ 115)
print(f"95th percentile: {norm_dist.ppf(0.95):.2f}")  # Inverse CDF
print()

# Generate random samples
samples_norm = norm_dist.rvs(size=1000)

# Plot distribution
x = np.linspace(50, 150, 200)
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(x, norm_dist.pdf(x), 'b-', linewidth=2, label='PDF')
plt.fill_between(x, norm_dist.pdf(x), alpha=0.3)
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.title('Normal Distribution PDF', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(x, norm_dist.cdf(x), 'r-', linewidth=2, label='CDF')
plt.xlabel('Value')
plt.ylabel('Cumulative Probability')
plt.title('Normal Distribution CDF', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.hist(samples_norm, bins=30, density=True, alpha=0.7, edgecolor='black', label='Samples')
plt.plot(x, norm_dist.pdf(x), 'r-', linewidth=2, label='Theoretical PDF')
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Random Samples vs Theory', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('normal_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Example 3: Other common distributions
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Poisson distribution
lambda_poisson = 5
x_poisson = np.arange(0, 20)
axes[0, 0].bar(x_poisson, stats.poisson.pmf(x_poisson, lambda_poisson), alpha=0.7)
axes[0, 0].set_title(f'Poisson(λ={lambda_poisson})', fontweight='bold')
axes[0, 0].set_xlabel('k')
axes[0, 0].set_ylabel('P(X=k)')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Exponential distribution
x_exp = np.linspace(0, 5, 100)
axes[0, 1].plot(x_exp, stats.expon.pdf(x_exp, scale=1), linewidth=2)
axes[0, 1].fill_between(x_exp, stats.expon.pdf(x_exp, scale=1), alpha=0.3)
axes[0, 1].set_title('Exponential(scale=1)', fontweight='bold')
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('PDF')
axes[0, 1].grid(True, alpha=0.3)

# Uniform distribution
x_unif = np.linspace(0, 10, 200)
axes[0, 2].plot(x_unif, stats.uniform.pdf(x_unif, loc=2, scale=5), linewidth=2)
axes[0, 2].fill_between(x_unif, stats.uniform.pdf(x_unif, loc=2, scale=5), alpha=0.3)
axes[0, 2].set_title('Uniform[2, 7]', fontweight='bold')
axes[0, 2].set_xlabel('x')
axes[0, 2].set_ylabel('PDF')
axes[0, 2].grid(True, alpha=0.3)

# Binomial distribution
n_trials, p_success = 20, 0.5
x_binom = np.arange(0, n_trials + 1)
axes[1, 0].bar(x_binom, stats.binom.pmf(x_binom, n_trials, p_success), alpha=0.7)
axes[1, 0].set_title(f'Binomial(n={n_trials}, p={p_success})', fontweight='bold')
axes[1, 0].set_xlabel('k')
axes[1, 0].set_ylabel('P(X=k)')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Chi-square distribution
x_chi2 = np.linspace(0, 20, 200)
for df in [2, 5, 10]:
    axes[1, 1].plot(x_chi2, stats.chi2.pdf(x_chi2, df), linewidth=2, label=f'df={df}')
axes[1, 1].set_title('Chi-square(df)', fontweight='bold')
axes[1, 1].set_xlabel('x')
axes[1, 1].set_ylabel('PDF')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Student's t distribution
x_t = np.linspace(-4, 4, 200)
for df in [1, 5, 30]:
    axes[1, 2].plot(x_t, stats.t.pdf(x_t, df), linewidth=2, label=f'df={df}')
axes[1, 2].plot(x_t, stats.norm.pdf(x_t), 'k--', linewidth=2, label='Normal')
axes[1, 2].set_title("Student's t(df)", fontweight='bold')
axes[1, 2].set_xlabel('x')
axes[1, 2].set_ylabel('PDF')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('distributions_overview.png', dpi=300, bbox_inches='tight')
plt.show()

# Example 4: Correlation and regression
x_corr = np.random.randn(100)
y_corr = 2 * x_corr + np.random.randn(100) * 0.5

# Pearson correlation
r_pearson, p_pearson = stats.pearsonr(x_corr, y_corr)
print("=== Correlation ===")
print(f"Pearson r: {r_pearson:.4f}, p-value: {p_pearson:.4e}")

# Spearman correlation (rank-based, robust to outliers)
r_spearman, p_spearman = stats.spearmanr(x_corr, y_corr)
print(f"Spearman ρ: {r_spearman:.4f}, p-value: {p_spearman:.4e}")
print()

# Linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(x_corr, y_corr)
print("=== Linear Regression ===")
print(f"Slope: {slope:.4f} ± {std_err:.4f}")
print(f"Intercept: {intercept:.4f}")
print(f"R-squared: {r_value**2:.4f}")
print(f"P-value: {p_value:.4e}")

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(x_corr, y_corr, alpha=0.6, s=50)
x_line = np.linspace(x_corr.min(), x_corr.max(), 100)
y_line = slope * x_line + intercept
plt.plot(x_line, y_line, 'r-', linewidth=2, 
         label=f'y = {slope:.2f}x + {intercept:.2f}')
plt.xlabel('X', fontsize=12)
plt.ylabel('Y', fontsize=12)
plt.title(f'Linear Regression (R² = {r_value**2:.4f})', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('linear_regression.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Code Explanation:

**Line 11:** `stats.ttest_ind(sample1, sample2)` performs independent t-test:
- Tests null hypothesis that two samples have equal means
- Returns t-statistic and p-value
- Use when comparing means of two independent groups

**Line 21:** `stats.ks_2samp(sample1, sample2)` performs Kolmogorov-Smirnov test:
- Tests if two samples come from same distribution
- More general than t-test (not just means)
- Non-parametric (no assumption about distribution shape)

**Line 31:** `stats.chisquare(observed, expected)` performs chi-square goodness-of-fit test:
- Tests if observed frequencies match expected frequencies
- Used for categorical data
- Returns χ² statistic and p-value

**Line 38:** `stats.norm(loc, scale)` creates normal distribution object:
- `loc` - mean (μ)
- `scale` - standard deviation (σ)
Returns distribution object with many methods

**Line 44:** `.pdf(x)` - probability density function, gives height of distribution at x

**Line 45:** `.cdf(x)` - cumulative distribution function, gives P(X ≤ x)

**Line 46:** `.ppf(q)` - percent point function (inverse CDF), gives x where P(X ≤ x) = q

**Line 50:** `.rvs(size)` - random variates, generates random samples from distribution

**Line 99:** `stats.poisson.pmf(k, lambda)` - probability mass function for discrete distribution
- PMF for discrete distributions (gives P(X = k))
- PDF for continuous distributions (gives density)

**Line 109:** `stats.expon.pdf(x, scale)` - exponential distribution
- `scale` = 1/λ (rate parameter)
- Models time between events

**Line 116:** `stats.uniform.pdf(x, loc, scale)` - uniform distribution
- `loc` - start of interval
- `scale` - width of interval
- Uniform on [loc, loc + scale]

**Line 124:** `stats.binom.pmf(k, n, p)` - binomial distribution
- `n` - number of trials
- `p` - success probability
- Models number of successes in n trials

**Line 131-132:** Multiple distributions plotted with different parameters to show shape changes

**Line 158:** `stats.pearsonr(x, y)` - Pearson correlation coefficient:
- Measures linear relationship
- r = 1: perfect positive correlation
- r = 0: no correlation
- r = -1: perfect negative correlation

**Line 162:** `stats.spearmanr(x, y)` - Spearman correlation:
- Based on ranks, not actual values
- More robust to outliers
- Detects monotonic relationships (not just linear)

**Line 167:** `stats.linregress(x, y)` performs simple linear regression:
- Fits line y = slope*x + intercept
- Returns: slope, intercept, r_value, p_value, std_err
- `r_value**2` is R-squared (goodness of fit)

---

## Example 5: Signal Processing

```python
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Generate noisy signal
np.random.seed(42)
t = np.linspace(0, 10, 500)
clean_signal = np.sin(2 * np.pi * 1 * t) + 0.5 * np.sin(2 * np.pi * 3 * t)
noisy_signal = clean_signal + 0.5 * np.random.randn(len(t))

print("=== Signal Processing ===")

# Savitzky-Golay filter (smoothing that preserves features)
window_length = 51  # Must be odd
polyorder = 3
smoothed_savgol = signal.savgol_filter(noisy_signal, window_length, polyorder)

# Moving average (simple smoothing)
window_size = 20
smoothed_moving = np.convolve(noisy_signal, np.ones(window_size)/window_size, mode='same')

# Median filter (removes spikes, preserves edges)
smoothed_median = signal.medfilt(noisy_signal, kernel_size=21)

# Plot comparison
plt.figure(figsize=(14, 8))

plt.subplot(2, 2, 1)
plt.plot(t, noisy_signal, 'gray', alpha=0.5, label='Noisy')
plt.plot(t, clean_signal, 'b--', label='True', linewidth=2)
plt.plot(t, smoothed_savgol, 'r-', label='Savitzky-Golay', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Signal')
plt.title('Savitzky-Golay Filter', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.plot(t, noisy_signal, 'gray', alpha=0.5, label='Noisy')
plt.plot(t, clean_signal, 'b--', label='True', linewidth=2)
plt.plot(t, smoothed_moving, 'g-', label='Moving Average', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Signal')
plt.title('Moving Average', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
plt.plot(t, noisy_signal, 'gray', alpha=0.5, label='Noisy')
plt.plot(t, clean_signal, 'b--', label='True', linewidth=2)
plt.plot(t, smoothed_median, 'm-', label='Median Filter', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Signal')
plt.title('Median Filter', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
plt.plot(t, smoothed_savgol, 'r-', alpha=0.7, label='Savitzky-Golay', linewidth=2)
plt.plot(t, smoothed_moving, 'g-', alpha=0.7, label='Moving Average', linewidth=2)
plt.plot(t, smoothed_median, 'm-', alpha=0.7, label='Median', linewidth=2)
plt.plot(t, clean_signal, 'b--', label='True', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Signal')
plt.title('All Filters Compared', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('signal_filtering.png', dpi=300, bbox_inches='tight')
plt.show()

# Peak detection
signal_with_peaks = 2 * np.sin(t) + 0.5 * np.random.randn(len(t))
peaks, properties = signal.find_peaks(signal_with_peaks, height=1.0, distance=20)

plt.figure(figsize=(12, 5))
plt.plot(t, signal_with_peaks, 'b-', label='Signal')
plt.plot(t[peaks], signal_with_peaks[peaks], 'ro', markersize=10, label='Peaks')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.title('Peak Detection', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('peak_detection.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Number of peaks found: {len(peaks)}")
print(f"Peak locations (indices): {peaks[:10]}...")  # First 10
print(f"Peak heights: {properties['peak_heights'][:10]}...")
print()

# Frequency analysis with FFT
sample_rate = 100  # Hz
duration = 4  # seconds
t_fft = np.linspace(0, duration, sample_rate * duration)

# Signal with 3 frequency components
sig_fft = (np.sin(2 * np.pi * 5 * t_fft) + 
           0.5 * np.sin(2 * np.pi * 10 * t_fft) + 
           0.3 * np.sin(2 * np.pi * 20 * t_fft))

# Compute FFT
fft_vals = np.fft.fft(sig_fft)
fft_freq = np.fft.fftfreq(len(sig_fft), 1/sample_rate)

# Only positive frequencies
positive_freq = fft_freq[:len(fft_freq)//2]
positive_fft = np.abs(fft_vals[:len(fft_vals)//2])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

ax1.plot(t_fft, sig_fft, 'b-', linewidth=1)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Amplitude')
ax1.set_title('Time Domain Signal', fontweight='bold')
ax1.grid(True, alpha=0.3)

ax2.plot(positive_freq, positive_fft, 'r-', linewidth=2)
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Magnitude')
ax2.set_title('Frequency Spectrum (FFT)', fontweight='bold')
ax2.set_xlim(0, 30)
ax2.grid(True, alpha=0.3)

# Mark peaks at 5, 10, 20 Hz
ax2.axvline(5, color='green', linestyle='--', alpha=0.5, label='5 Hz component')
ax2.axvline(10, color='green', linestyle='--', alpha=0.5, label='10 Hz component')
ax2.axvline(20, color='green', linestyle='--', alpha=0.5, label='20 Hz component')
ax2.legend()

plt.tight_layout()
plt.savefig('fft_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("FFT analysis complete - peaks should appear at 5, 10, and 20 Hz")
```

### Code Explanation:

**Line 15:** `signal.savgol_filter(data, window_length, polyorder)` applies Savitzky-Golay filter:
- `window_length` - size of filter window (must be odd)
- `polyorder` - polynomial order for fitting
- Smooths while preserving peaks better than moving average
- Good for spectroscopy, noisy derivatives

**Line 19:** Moving average using convolution:
- `np.ones(window_size)/window_size` creates averaging kernel
- `mode='same'` keeps output same size as input
- Simple but can blur sharp features

**Line 22:** `signal.medfilt(data, kernel_size)` applies median filter:
- Replaces each point with median of surrounding window
- Excellent at removing spikes/outliers
- Preserves edges better than averaging

**Line 77:** `signal.find_peaks(data, height, distance)` detects peaks:
- `height` - minimum peak height
- `distance` - minimum distance between peaks (in samples)
- `prominence` - how much peak stands out from surroundings
- Returns indices of peaks and dictionary of properties

**Line 85:** `properties['peak_heights']` contains the y-values at detected peaks

**Line 99-101:** Create signal with multiple frequency components to demonstrate FFT

**Line 104:** `np.fft.fft(signal)` computes Fast Fourier Transform:
- Converts time domain → frequency domain
- Returns complex numbers (use `np.abs()` for magnitude)

**Line 105:** `np.fft.fftfreq(n, d)` generates frequency values:
- `n` - number of samples
- `d` - sample spacing (1/sample_rate)

**Line 108-109:** FFT output is symmetric, so we only need positive frequencies (first half)

FFT reveals which frequencies are present in signal - peaks at 5, 10, 20 Hz match our input components.

---

## Key SciPy Concepts Summary

### Module Organization
```python
from scipy import optimize    # Optimization, root finding
from scipy import integrate   # Integration, ODEs
from scipy import interpolate # Interpolation, smoothing
from scipy import stats       # Statistics, distributions
from scipy import signal      # Signal processing
from scipy import linalg      # Linear algebra (advanced)
from scipy import sparse      # Sparse matrices
from scipy import spatial     # Spatial algorithms
```

### Common Workflows

**Curve Fitting:**
1. Define model function
2. Provide initial guess
3. Call `curve_fit()`
4. Extract uncertainties from covariance matrix

**Solving ODEs:**
1. Define ODE as `f(t, y)` returning derivatives
2. Set initial conditions
3. Call `solve_ivp()`
4. Extract solution from result object

**Interpolation:**
1. Create interpolation function with data
2. Call function with new x values
3. Don't extrapolate far beyond data

**Statistical Testing:**
1. Choose appropriate test (t-test, KS test, etc.)
2. Check assumptions (normality, independence)
3. Interpret p-value (< 0.05 typically significant)

### When to Use What

**Optimization Methods:**
- BFGS: General purpose, gradient-based
- Nelder-Mead: Derivative-free, slower
- Powell: Derivative-free, good for noisy functions
- L-BFGS-B: Constrained optimization with bounds

**Interpolation Types:**
- Linear: Fast, continuous but not smooth
- Quadratic: Smooth, can oscillate
- Cubic: Smooth, standard choice
- Splines: Flexible, smoothing option available

**Statistical Tests:**
- T-test: Compare means of two groups
- ANOVA: Compare means of multiple groups
- KS test: Compare distributions
- Chi-square: Categorical data, goodness of fit
- Pearson: Linear correlation
- Spearman: Monotonic correlation

**Filters:**
- Savitzky-Golay: Preserves peaks, good for derivatives
- Moving average: Simple, blurs features
- Median: Removes spikes, preserves edges
- Butterworth: Frequency-selective filtering

## Common Gotchas

1. **Curve fitting convergence**: Provide good initial guesses (`p0`). Bad guesses = failed fits.

2. **ODE function signature**: Must be `f(t, y)` not `f(y, t)`. Order matters!

3. **Interpolation extrapolation**: Don't trust values outside data range. Use `fill_value` parameter to control behavior.

4. **P-value misinterpretation**: p < 0.05 doesn't mean large effect, just statistical significance.

5. **FFT frequency units**: Match your sample rate. If sampling at 100 Hz, frequencies are in Hz.

6. **Integration accuracy**: `quad()` is usually fine, but check error estimate. For oscillatory functions, may need specialized methods.

---

## Practice Exercises

1. Fit a power law model `y = A * x^b` to data points. Extract parameter uncertainties.

2. Solve the damped harmonic oscillator ODE: `d²y/dt² + 2ζω₀ dy/dt + ω₀²y = 0`

3. Interpolate sparse 2D data on a fine grid and create a contour plot.

4. Generate 1000 samples from chi-square distribution, perform KS test against theoretical distribution.

5. Create a noisy signal with 3 frequency components, apply FFT, identify the frequencies.

## Additional Resources

- Official Documentation: https://docs.scipy.org/doc/scipy/
- SciPy Cookbook: https://scipy-cookbook.readthedocs.io/
- Stats Tutorial: https://docs.scipy.org/doc/scipy/tutorial/stats.html
- Optimization Guide: https://docs.scipy.org/doc/scipy/tutorial/optimize.html
SCIPY_EOF
cat /home/claude/scipy_complete_guide.md
Output

# SciPy - Complete Guide

## Introduction

SciPy (Scientific Python) is a collection of mathematical algorithms and convenience functions built on NumPy. It provides high-level scientific and technical computing capabilities including optimization, integration, interpolation, signal processing, linear algebra, statistics, and more.

**Key Capabilities:**
- Optimization and root finding
- Integration and differential equations
- Interpolation
- Signal processing and filtering
- Statistical functions and distributions
- Linear algebra (advanced operations)
- Sparse matrices
- Spatial data structures and algorithms

**Why SciPy?**
- Built on NumPy (fast and efficient)
- Well-tested, reliable implementations
- Comprehensive scientific algorithms
- Standard tool in scientific Python

**Installation:**
```bash
pip install scipy
```

---

## Example 1: Curve Fitting and Optimization

```python
import numpy as np
from scipy.optimize import curve_fit, minimize
import matplotlib.pyplot as plt

# Generate synthetic data with noise
np.random.seed(42)
x_data = np.linspace(0, 10, 50)
y_true = 2.5 * np.exp(-0.5 * x_data) + 1.0
y_data = y_true + np.random.normal(0, 0.2, len(x_data))

# Define model function
def exponential_model(x, a, b, c):
    """Model: y = a * exp(-b * x) + c"""
    return a * np.exp(-b * x) + c

# Fit the model to data
initial_guess = [2.0, 0.5, 1.0]
params, covariance = curve_fit(exponential_model, x_data, y_data, p0=initial_guess)
a_fit, b_fit, c_fit = params

# Calculate parameter uncertainties
param_errors = np.sqrt(np.diag(covariance))

print("=== Curve Fitting Results ===")
print(f"Parameter a: {a_fit:.4f} ± {param_errors[0]:.4f}")
print(f"Parameter b: {b_fit:.4f} ± {param_errors[1]:.4f}")
print(f"Parameter c: {c_fit:.4f} ± {param_errors[2]:.4f}")
print()

# Calculate R-squared
y_pred = exponential_model(x_data, a_fit, b_fit, c_fit)
ss_res = np.sum((y_data - y_pred) ** 2)
ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
r_squared = 1 - (ss_res / ss_tot)
print(f"R-squared: {r_squared:.4f}")
print()

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, label='Data', alpha=0.6, s=50)
plt.plot(x_data, y_true, 'g--', label='True function', linewidth=2)
x_smooth = np.linspace(0, 10, 200)
y_fit = exponential_model(x_smooth, a_fit, b_fit, c_fit)
plt.plot(x_smooth, y_fit, 'r-', label='Fitted function', linewidth=2)
plt.xlabel('X', fontsize=12)
plt.ylabel('Y', fontsize=12)
plt.title('Exponential Curve Fitting', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('curve_fitting.png', dpi=300, bbox_inches='tight')
plt.show()

# Optimization example: Find minimum of a function
def rosenbrock(x):
    """Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2"""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

# Starting point
x0 = np.array([0, 0])

# Minimize the function
result = minimize(rosenbrock, x0, method='BFGS')

print("=== Optimization Results ===")
print(f"Minimum found at: x = {result.x[0]:.6f}, y = {result.x[1]:.6f}")
print(f"Function value at minimum: {result.fun:.10f}")
print(f"Number of iterations: {result.nit}")
print(f"Success: {result.success}")
```

### Code Explanation:

**Line 7-9:** Generate synthetic data with noise to demonstrate curve fitting. The true function is an exponential decay plus a constant.

**Line 12-14:** Define the model function we want to fit. It takes x and three parameters (a, b, c) and returns the predicted y values according to the model equation.

**Line 17:** `curve_fit(func, xdata, ydata, p0)` finds optimal parameters:
- `func` - the model function
- `xdata`, `ydata` - the data to fit
- `p0` - initial guess for parameters (helps convergence)
Returns: optimal parameters and covariance matrix

**Line 21:** `covariance` is a matrix describing parameter uncertainties and correlations. Diagonal elements are variances of parameters.

**Line 22:** `np.sqrt(np.diag(covariance))` extracts standard errors (uncertainties) of each parameter from the diagonal of covariance matrix.

**Line 32-34:** Calculate R-squared (coefficient of determination):
- `ss_res` - sum of squared residuals (prediction errors)
- `ss_tot` - total sum of squares (variance in data)
- R² = 1 means perfect fit, R² = 0 means model no better than mean

**Line 56-58:** Define Rosenbrock function, a common test function for optimization. Has global minimum at (1, 1).

**Line 64:** `minimize(func, x0, method)` finds minimum of function:
- `func` - function to minimize
- `x0` - starting point for search
- `method` - optimization algorithm ('BFGS', 'Nelder-Mead', 'Powell', etc.)

**Line 67:** `result.x` contains the optimal parameters found.

**Line 68:** `result.fun` is the function value at the minimum.

**Line 69:** `result.nit` is the number of iterations taken.

**Line 70:** `result.success` indicates whether optimization converged successfully.

---

## Example 2: Integration and Differential Equations

```python
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

# Example 1: Definite integral
def gaussian(x):
    """Gaussian function"""
    return np.exp(-x**2)

# Integrate from -infinity to infinity
result, error = integrate.quad(gaussian, -np.inf, np.inf)
print("=== Numerical Integration ===")
print(f"∫ exp(-x²) dx from -∞ to ∞ = {result:.10f}")
print(f"Analytical result: √π = {np.sqrt(np.pi):.10f}")
print(f"Integration error estimate: {error:.2e}")
print()

# Example 2: Integrate discrete data (trapezoid rule)
x_discrete = np.linspace(0, 10, 100)
y_discrete = np.sin(x_discrete) * np.exp(-0.1 * x_discrete)
area = integrate.trapezoid(y_discrete, x_discrete)
print(f"Area under curve (trapezoid): {area:.6f}")
print()

# Example 3: Solve ordinary differential equation (ODE)
# Solve: dy/dt = -0.5*y, y(0) = 10
def exponential_decay(t, y):
    """ODE: dy/dt = -0.5*y"""
    return -0.5 * y

# Initial condition and time span
y0 = [10]
t_span = (0, 10)
t_eval = np.linspace(0, 10, 100)

# Solve ODE
solution = integrate.solve_ivp(exponential_decay, t_span, y0, t_eval=t_eval)

# Analytical solution for comparison
y_analytical = 10 * np.exp(-0.5 * t_eval)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(solution.t, solution.y[0], 'b-', label='Numerical solution', linewidth=2)
plt.plot(t_eval, y_analytical, 'r--', label='Analytical solution', linewidth=2)
plt.xlabel('Time', fontsize=12)
plt.ylabel('y(t)', fontsize=12)
plt.title('Solving ODE: dy/dt = -0.5y, y(0) = 10', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ode_solution.png', dpi=300, bbox_inches='tight')
plt.show()

# Example 4: System of ODEs (Lotka-Volterra predator-prey)
def lotka_volterra(t, y):
    """Predator-prey equations"""
    prey, predator = y
    alpha = 1.5    # prey birth rate
    beta = 1.0     # predation rate
    delta = 0.75   # predator efficiency
    gamma = 1.0    # predator death rate
    
    dprey_dt = alpha * prey - beta * prey * predator
    dpredator_dt = delta * prey * predator - gamma * predator
    
    return [dprey_dt, dpredator_dt]

# Initial populations
y0_system = [2.0, 1.0]  # [prey, predator]
t_span_system = (0, 20)
t_eval_system = np.linspace(0, 20, 1000)

# Solve system
solution_system = integrate.solve_ivp(lotka_volterra, t_span_system, y0_system, 
                                     t_eval=t_eval_system, method='RK45')

# Plot phase space and time series
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Time series
ax1.plot(solution_system.t, solution_system.y[0], 'b-', label='Prey', linewidth=2)
ax1.plot(solution_system.t, solution_system.y[1], 'r-', label='Predator', linewidth=2)
ax1.set_xlabel('Time', fontsize=12)
ax1.set_ylabel('Population', fontsize=12)
ax1.set_title('Lotka-Volterra: Population Dynamics', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Phase space
ax2.plot(solution_system.y[0], solution_system.y[1], 'g-', linewidth=2)
ax2.plot(solution_system.y[0][0], solution_system.y[1][0], 'go', markersize=10, 
         label='Start')
ax2.set_xlabel('Prey Population', fontsize=12)
ax2.set_ylabel('Predator Population', fontsize=12)
ax2.set_title('Phase Space', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lotka_volterra.png', dpi=300, bbox_inches='tight')
plt.show()

print("Lotka-Volterra system solved successfully")
```

### Code Explanation:

**Line 6-8:** Define a function to integrate. Must take single variable as input.

**Line 11:** `integrate.quad(func, a, b)` performs numerical integration (quadrature):
- `func` - function to integrate
- `a, b` - integration limits (can use `np.inf` for infinity)
Returns: (result, error_estimate)

Uses adaptive algorithm that automatically refines where needed for accuracy.

**Line 21:** `integrate.trapezoid(y, x)` integrates discrete data points using trapezoid rule:
- `y` - function values at points
- `x` - x-coordinates of points
Approximates area by connecting points with straight lines and summing trapezoid areas.

**Line 26-28:** Define ODE as function `f(t, y)` that returns dy/dt. SciPy format requires this specific signature.

**Line 31-33:** Set up problem:
- `y0` - initial condition(s) as list
- `t_span` - tuple (t_start, t_end) for integration
- `t_eval` - specific times where solution is needed (optional)

**Line 36:** `integrate.solve_ivp(func, t_span, y0, t_eval)` solves initial value problem:
- `func` - ODE function
- `t_span` - integration interval
- `y0` - initial values
- `t_eval` - times to evaluate solution
- `method` - solver ('RK45', 'RK23', 'DOP853', 'Radau', 'BDF')

Returns solution object with `.t` (times) and `.y` (solution values).

**Line 60-68:** Define system of ODEs. Function returns list with derivative of each variable. For predator-prey:
- Prey grows exponentially, reduced by predation
- Predators grow when eating prey, die naturally

**Line 71:** Initial conditions for system must match number of equations (2 populations).

**Line 76-77:** `solve_ivp` handles systems automatically - just provide list of initial conditions and return list of derivatives.

**Line 87:** `solution.y[0]` accesses first variable (prey), `solution.y[1]` accesses second (predator).

**Line 94:** Phase space plot shows predator vs prey population, revealing cyclic dynamics.

---

## Example 3: Interpolation and Smoothing

```python
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

# Generate sparse data with noise
np.random.seed(42)
x_sparse = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_sparse = np.sin(x_sparse) + 0.1 * np.random.randn(len(x_sparse))

# Create different interpolation functions
f_linear = interpolate.interp1d(x_sparse, y_sparse, kind='linear')
f_quadratic = interpolate.interp1d(x_sparse, y_sparse, kind='quadratic')
f_cubic = interpolate.interp1d(x_sparse, y_sparse, kind='cubic')

# Create smooth x values for plotting
x_smooth = np.linspace(0, 10, 200)

# Evaluate interpolations
y_linear = f_linear(x_smooth)
y_quadratic = f_quadratic(x_smooth)
y_cubic = f_cubic(x_smooth)

# Plot comparison
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(x_sparse, y_sparse, 'o', label='Data points', markersize=8)
plt.plot(x_smooth, y_linear, '-', label='Linear', linewidth=2)
plt.title('Linear Interpolation', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.plot(x_sparse, y_sparse, 'o', label='Data points', markersize=8)
plt.plot(x_smooth, y_quadratic, '-', label='Quadratic', linewidth=2)
plt.title('Quadratic Interpolation', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
plt.plot(x_sparse, y_sparse, 'o', label='Data points', markersize=8)
plt.plot(x_smooth, y_cubic, '-', label='Cubic', linewidth=2)
plt.title('Cubic Interpolation', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
plt.plot(x_sparse, y_sparse, 'ko', label='Data', markersize=8, zorder=3)
plt.plot(x_smooth, y_linear, '-', alpha=0.5, label='Linear', linewidth=2)
plt.plot(x_smooth, y_quadratic, '-', alpha=0.5, label='Quadratic', linewidth=2)
plt.plot(x_smooth, y_cubic, '-', alpha=0.5, label='Cubic', linewidth=2)
plt.title('All Methods Compared', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('interpolation_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 2D interpolation example
x_2d = np.linspace(0, 3, 4)
y_2d = np.linspace(0, 3, 4)
z_2d = np.array([[1, 2, 3, 4],
                 [2, 3, 4, 5],
                 [3, 4, 5, 6],
                 [4, 5, 6, 7]])

# Create 2D interpolation function
f_2d = interpolate.interp2d(x_2d, y_2d, z_2d, kind='cubic')

# Evaluate on finer grid
x_fine = np.linspace(0, 3, 20)
y_fine = np.linspace(0, 3, 20)
z_fine = f_2d(x_fine, y_fine)

# Plot 2D interpolation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Original data
im1 = ax1.imshow(z_2d, extent=[0, 3, 0, 3], origin='lower', cmap='viridis')
ax1.set_title('Original 4x4 Data', fontweight='bold')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
plt.colorbar(im1, ax=ax1)

# Interpolated data
im2 = ax2.imshow(z_fine, extent=[0, 3, 0, 3], origin='lower', cmap='viridis')
ax2.set_title('Interpolated 20x20 Data', fontweight='bold')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
plt.colorbar(im2, ax=ax2)

plt.tight_layout()
plt.savefig('2d_interpolation.png', dpi=300, bbox_inches='tight')
plt.show()

# Univariate spline (smoothing)
x_noisy = np.linspace(0, 10, 50)
y_noisy = np.sin(x_noisy) + 0.3 * np.random.randn(len(x_noisy))

# Fit spline with different smoothing factors
spline_smooth0 = interpolate.UnivariateSpline(x_noisy, y_noisy, s=0)  # No smoothing
spline_smooth1 = interpolate.UnivariateSpline(x_noisy, y_noisy, s=1)  # Some smoothing
spline_smooth5 = interpolate.UnivariateSpline(x_noisy, y_noisy, s=5)  # More smoothing

x_dense = np.linspace(0, 10, 500)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(x_noisy, y_noisy, 'o', alpha=0.5, label='Noisy data')
plt.plot(x_dense, spline_smooth0(x_dense), 'r-', label='s=0 (interpolate)', linewidth=2)
plt.plot(x_dense, np.sin(x_dense), 'g--', label='True function', linewidth=2)
plt.title('Smoothing s=0', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(x_noisy, y_noisy, 'o', alpha=0.5, label='Noisy data')
plt.plot(x_dense, spline_smooth1(x_dense), 'r-', label='s=1', linewidth=2)
plt.plot(x_dense, np.sin(x_dense), 'g--', label='True function', linewidth=2)
plt.title('Smoothing s=1', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.plot(x_noisy, y_noisy, 'o', alpha=0.5, label='Noisy data')
plt.plot(x_dense, spline_smooth5(x_dense), 'r-', label='s=5', linewidth=2)
plt.plot(x_dense, np.sin(x_dense), 'g--', label='True function', linewidth=2)
plt.title('Smoothing s=5', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('spline_smoothing.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Code Explanation:

**Line 11-13:** `interpolate.interp1d(x, y, kind)` creates interpolation function:
- `x, y` - known data points
- `kind` - interpolation type: 'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'
Returns a function that can be called with new x values.

**Line 16:** Create fine grid of x values to evaluate interpolation and produce smooth curves.

**Line 19-21:** Call interpolation functions like regular functions: `f(x_new)` returns interpolated y values.

**Line 66:** `interpolate.interp2d(x, y, z, kind)` creates 2D interpolation function:
- `x, y` - 1D coordinate arrays
- `z` - 2D array of function values
- `kind` - 'linear', 'cubic', 'quintic'
Returns function that takes new x and y arrays.

**Line 71:** Call 2D interpolation: `f_2d(x_new, y_new)` returns 2D array of interpolated values.

**Line 105-107:** `interpolate.UnivariateSpline(x, y, s)` fits smoothing spline:
- `x, y` - data points
- `s` - smoothing factor:
  - `s=0` passes through all points (no smoothing)
  - `s>0` allows deviation from points (more smoothing)
  - Larger s = smoother curve

Returns callable spline object that can compute values and derivatives.

**Interpolation Notes:**
- Linear: connects points with straight lines (C0 continuity)
- Quadratic: uses parabolas (C1 continuity)
- Cubic: uses cubic polynomials (C2 continuity - smooth derivatives)
- Don't extrapolate far beyond data range - unreliable

---

## Example 4: Statistical Functions and Distributions

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Example 1: Statistical tests
# Generate two samples
np.random.seed(42)
sample1 = np.random.normal(100, 15, 100)
sample2 = np.random.normal(105, 15, 100)

print("=== Statistical Tests ===")

# T-test: Are means significantly different?
t_stat, p_value_ttest = stats.ttest_ind(sample1, sample2)
print(f"T-test: t={t_stat:.4f}, p-value={p_value_ttest:.4f}")
if p_value_ttest < 0.05:
    print("  → Samples have significantly different means (p < 0.05)")
else:
    print("  → No significant difference in means (p ≥ 0.05)")
print()

# Kolmogorov-Smirnov test: Are distributions different?
ks_stat, p_value_ks = stats.ks_2samp(sample1, sample2)
print(f"KS test: statistic={ks_stat:.4f}, p-value={p_value_ks:.4f}")
if p_value_ks < 0.05:
    print("  → Samples from different distributions (p < 0.05)")
else:
    print("  → Samples from same distribution (p ≥ 0.05)")
print()

# Chi-square test for goodness of fit
observed = np.array([15, 25, 35, 25])
expected = np.array([20, 25, 30, 25])
chi2_stat, p_value_chi2 = stats.chisquare(observed, expected)
print(f"Chi-square test: χ²={chi2_stat:.4f}, p-value={p_value_chi2:.4f}")
print()

# Example 2: Probability distributions
# Normal distribution
print("=== Normal Distribution ===")
mu, sigma = 100, 15
norm_dist = stats.norm(loc=mu, scale=sigma)

print(f"Mean: {norm_dist.mean()}")
print(f"Std: {norm_dist.std()}")
print(f"PDF at x=100: {norm_dist.pdf(100):.6f}")
print(f"CDF at x=115: {norm_dist.cdf(115):.4f}")  # P(X ≤ 115)
print(f"95th percentile: {norm_dist.ppf(0.95):.2f}")  # Inverse CDF
print()

# Generate random samples
samples_norm = norm_dist.rvs(size=1000)

# Plot distribution
x = np.linspace(50, 150, 200)
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(x, norm_dist.pdf(x), 'b-', linewidth=2, label='PDF')
plt.fill_between(x, norm_dist.pdf(x), alpha=0.3)
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.title('Normal Distribution PDF', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(x, norm_dist.cdf(x), 'r-', linewidth=2, label='CDF')
plt.xlabel('Value')
plt.ylabel('Cumulative Probability')
plt.title('Normal Distribution CDF', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.hist(samples_norm, bins=30, density=True, alpha=0.7, edgecolor='black', label='Samples')
plt.plot(x, norm_dist.pdf(x), 'r-', linewidth=2, label='Theoretical PDF')
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Random Samples vs Theory', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('normal_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Example 3: Other common distributions
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Poisson distribution
lambda_poisson = 5
x_poisson = np.arange(0, 20)
axes[0, 0].bar(x_poisson, stats.poisson.pmf(x_poisson, lambda_poisson), alpha=0.7)
axes[0, 0].set_title(f'Poisson(λ={lambda_poisson})', fontweight='bold')
axes[0, 0].set_xlabel('k')
axes[0, 0].set_ylabel('P(X=k)')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Exponential distribution
x_exp = np.linspace(0, 5, 100)
axes[0, 1].plot(x_exp, stats.expon.pdf(x_exp, scale=1), linewidth=2)
axes[0, 1].fill_between(x_exp, stats.expon.pdf(x_exp, scale=1), alpha=0.3)
axes[0, 1].set_title('Exponential(scale=1)', fontweight='bold')
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('PDF')
axes[0, 1].grid(True, alpha=0.3)

# Uniform distribution
x_unif = np.linspace(0, 10, 200)
axes[0, 2].plot(x_unif, stats.uniform.pdf(x_unif, loc=2, scale=5), linewidth=2)
axes[0, 2].fill_between(x_unif, stats.uniform.pdf(x_unif, loc=2, scale=5), alpha=0.3)
axes[0, 2].set_title('Uniform[2, 7]', fontweight='bold')
axes[0, 2].set_xlabel('x')
axes[0, 2].set_ylabel('PDF')
axes[0, 2].grid(True, alpha=0.3)

# Binomial distribution
n_trials, p_success = 20, 0.5
x_binom = np.arange(0, n_trials + 1)
axes[1, 0].bar(x_binom, stats.binom.pmf(x_binom, n_trials, p_success), alpha=0.7)
axes[1, 0].set_title(f'Binomial(n={n_trials}, p={p_success})', fontweight='bold')
axes[1, 0].set_xlabel('k')
axes[1, 0].set_ylabel('P(X=k)')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Chi-square distribution
x_chi2 = np.linspace(0, 20, 200)
for df in [2, 5, 10]:
    axes[1, 1].plot(x_chi2, stats.chi2.pdf(x_chi2, df), linewidth=2, label=f'df={df}')
axes[1, 1].set_title('Chi-square(df)', fontweight='bold')
axes[1, 1].set_xlabel('x')
axes[1, 1].set_ylabel('PDF')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Student's t distribution
x_t = np.linspace(-4, 4, 200)
for df in [1, 5, 30]:
    axes[1, 2].plot(x_t, stats.t.pdf(x_t, df), linewidth=2, label=f'df={df}')
axes[1, 2].plot(x_t, stats.norm.pdf(x_t), 'k--', linewidth=2, label='Normal')
axes[1, 2].set_title("Student's t(df)", fontweight='bold')
axes[1, 2].set_xlabel('x')
axes[1, 2].set_ylabel('PDF')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('distributions_overview.png', dpi=300, bbox_inches='tight')
plt.show()

# Example 4: Correlation and regression
x_corr = np.random.randn(100)
y_corr = 2 * x_corr + np.random.randn(100) * 0.5

# Pearson correlation
r_pearson, p_pearson = stats.pearsonr(x_corr, y_corr)
print("=== Correlation ===")
print(f"Pearson r: {r_pearson:.4f}, p-value: {p_pearson:.4e}")

# Spearman correlation (rank-based, robust to outliers)
r_spearman, p_spearman = stats.spearmanr(x_corr, y_corr)
print(f"Spearman ρ: {r_spearman:.4f}, p-value: {p_spearman:.4e}")
print()

# Linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(x_corr, y_corr)
print("=== Linear Regression ===")
print(f"Slope: {slope:.4f} ± {std_err:.4f}")
print(f"Intercept: {intercept:.4f}")
print(f"R-squared: {r_value**2:.4f}")
print(f"P-value: {p_value:.4e}")

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(x_corr, y_corr, alpha=0.6, s=50)
x_line = np.linspace(x_corr.min(), x_corr.max(), 100)
y_line = slope * x_line + intercept
plt.plot(x_line, y_line, 'r-', linewidth=2, 
         label=f'y = {slope:.2f}x + {intercept:.2f}')
plt.xlabel('X', fontsize=12)
plt.ylabel('Y', fontsize=12)
plt.title(f'Linear Regression (R² = {r_value**2:.4f})', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('linear_regression.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Code Explanation:

**Line 11:** `stats.ttest_ind(sample1, sample2)` performs independent t-test:
- Tests null hypothesis that two samples have equal means
- Returns t-statistic and p-value
- Use when comparing means of two independent groups

**Line 21:** `stats.ks_2samp(sample1, sample2)` performs Kolmogorov-Smirnov test:
- Tests if two samples come from same distribution
- More general than t-test (not just means)
- Non-parametric (no assumption about distribution shape)

**Line 31:** `stats.chisquare(observed, expected)` performs chi-square goodness-of-fit test:
- Tests if observed frequencies match expected frequencies
- Used for categorical data
- Returns χ² statistic and p-value

**Line 38:** `stats.norm(loc, scale)` creates normal distribution object:
- `loc` - mean (μ)
- `scale` - standard deviation (σ)
Returns distribution object with many methods

**Line 44:** `.pdf(x)` - probability density function, gives height of distribution at x

**Line 45:** `.cdf(x)` - cumulative distribution function, gives P(X ≤ x)

**Line 46:** `.ppf(q)` - percent point function (inverse CDF), gives x where P(X ≤ x) = q

**Line 50:** `.rvs(size)` - random variates, generates random samples from distribution

**Line 99:** `stats.poisson.pmf(k, lambda)` - probability mass function for discrete distribution
- PMF for discrete distributions (gives P(X = k))
- PDF for continuous distributions (gives density)

**Line 109:** `stats.expon.pdf(x, scale)` - exponential distribution
- `scale` = 1/λ (rate parameter)
- Models time between events

**Line 116:** `stats.uniform.pdf(x, loc, scale)` - uniform distribution
- `loc` - start of interval
- `scale` - width of interval
- Uniform on [loc, loc + scale]

**Line 124:** `stats.binom.pmf(k, n, p)` - binomial distribution
- `n` - number of trials
- `p` - success probability
- Models number of successes in n trials

**Line 131-132:** Multiple distributions plotted with different parameters to show shape changes

**Line 158:** `stats.pearsonr(x, y)` - Pearson correlation coefficient:
- Measures linear relationship
- r = 1: perfect positive correlation
- r = 0: no correlation
- r = -1: perfect negative correlation

**Line 162:** `stats.spearmanr(x, y)` - Spearman correlation:
- Based on ranks, not actual values
- More robust to outliers
- Detects monotonic relationships (not just linear)

**Line 167:** `stats.linregress(x, y)` performs simple linear regression:
- Fits line y = slope*x + intercept
- Returns: slope, intercept, r_value, p_value, std_err
- `r_value**2` is R-squared (goodness of fit)

---

## Example 5: Signal Processing

```python
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Generate noisy signal
np.random.seed(42)
t = np.linspace(0, 10, 500)
clean_signal = np.sin(2 * np.pi * 1 * t) + 0.5 * np.sin(2 * np.pi * 3 * t)
noisy_signal = clean_signal + 0.5 * np.random.randn(len(t))

print("=== Signal Processing ===")

# Savitzky-Golay filter (smoothing that preserves features)
window_length = 51  # Must be odd
polyorder = 3
smoothed_savgol = signal.savgol_filter(noisy_signal, window_length, polyorder)

# Moving average (simple smoothing)
window_size = 20
smoothed_moving = np.convolve(noisy_signal, np.ones(window_size)/window_size, mode='same')

# Median filter (removes spikes, preserves edges)
smoothed_median = signal.medfilt(noisy_signal, kernel_size=21)

# Plot comparison
plt.figure(figsize=(14, 8))

plt.subplot(2, 2, 1)
plt.plot(t, noisy_signal, 'gray', alpha=0.5, label='Noisy')
plt.plot(t, clean_signal, 'b--', label='True', linewidth=2)
plt.plot(t, smoothed_savgol, 'r-', label='Savitzky-Golay', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Signal')
plt.title('Savitzky-Golay Filter', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.plot(t, noisy_signal, 'gray', alpha=0.5, label='Noisy')
plt.plot(t, clean_signal, 'b--', label='True', linewidth=2)
plt.plot(t, smoothed_moving, 'g-', label='Moving Average', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Signal')
plt.title('Moving Average', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
plt.plot(t, noisy_signal, 'gray', alpha=0.5, label='Noisy')
plt.plot(t, clean_signal, 'b--', label='True', linewidth=2)
plt.plot(t, smoothed_median, 'm-', label='Median Filter', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Signal')
plt.title('Median Filter', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
plt.plot(t, smoothed_savgol, 'r-', alpha=0.7, label='Savitzky-Golay', linewidth=2)
plt.plot(t, smoothed_moving, 'g-', alpha=0.7, label='Moving Average', linewidth=2)
plt.plot(t, smoothed_median, 'm-', alpha=0.7, label='Median', linewidth=2)
plt.plot(t, clean_signal, 'b--', label='True', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Signal')
plt.title('All Filters Compared', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('signal_filtering.png', dpi=300, bbox_inches='tight')
plt.show()

# Peak detection
signal_with_peaks = 2 * np.sin(t) + 0.5 * np.random.randn(len(t))
peaks, properties = signal.find_peaks(signal_with_peaks, height=1.0, distance=20)

plt.figure(figsize=(12, 5))
plt.plot(t, signal_with_peaks, 'b-', label='Signal')
plt.plot(t[peaks], signal_with_peaks[peaks], 'ro', markersize=10, label='Peaks')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.title('Peak Detection', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('peak_detection.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Number of peaks found: {len(peaks)}")
print(f"Peak locations (indices): {peaks[:10]}...")  # First 10
print(f"Peak heights: {properties['peak_heights'][:10]}...")
print()

# Frequency analysis with FFT
sample_rate = 100  # Hz
duration = 4  # seconds
t_fft = np.linspace(0, duration, sample_rate * duration)

# Signal with 3 frequency components
sig_fft = (np.sin(2 * np.pi * 5 * t_fft) + 
           0.5 * np.sin(2 * np.pi * 10 * t_fft) + 
           0.3 * np.sin(2 * np.pi * 20 * t_fft))

# Compute FFT
fft_vals = np.fft.fft(sig_fft)
fft_freq = np.fft.fftfreq(len(sig_fft), 1/sample_rate)

# Only positive frequencies
positive_freq = fft_freq[:len(fft_freq)//2]
positive_fft = np.abs(fft_vals[:len(fft_vals)//2])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

ax1.plot(t_fft, sig_fft, 'b-', linewidth=1)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Amplitude')
ax1.set_title('Time Domain Signal', fontweight='bold')
ax1.grid(True, alpha=0.3)

ax2.plot(positive_freq, positive_fft, 'r-', linewidth=2)
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Magnitude')
ax2.set_title('Frequency Spectrum (FFT)', fontweight='bold')
ax2.set_xlim(0, 30)
ax2.grid(True, alpha=0.3)

# Mark peaks at 5, 10, 20 Hz
ax2.axvline(5, color='green', linestyle='--', alpha=0.5, label='5 Hz component')
ax2.axvline(10, color='green', linestyle='--', alpha=0.5, label='10 Hz component')
ax2.axvline(20, color='green', linestyle='--', alpha=0.5, label='20 Hz component')
ax2.legend()

plt.tight_layout()
plt.savefig('fft_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("FFT analysis complete - peaks should appear at 5, 10, and 20 Hz")
```

### Code Explanation:

**Line 15:** `signal.savgol_filter(data, window_length, polyorder)` applies Savitzky-Golay filter:
- `window_length` - size of filter window (must be odd)
- `polyorder` - polynomial order for fitting
- Smooths while preserving peaks better than moving average
- Good for spectroscopy, noisy derivatives

**Line 19:** Moving average using convolution:
- `np.ones(window_size)/window_size` creates averaging kernel
- `mode='same'` keeps output same size as input
- Simple but can blur sharp features

**Line 22:** `signal.medfilt(data, kernel_size)` applies median filter:
- Replaces each point with median of surrounding window
- Excellent at removing spikes/outliers
- Preserves edges better than averaging

**Line 77:** `signal.find_peaks(data, height, distance)` detects peaks:
- `height` - minimum peak height
- `distance` - minimum distance between peaks (in samples)
- `prominence` - how much peak stands out from surroundings
- Returns indices of peaks and dictionary of properties

**Line 85:** `properties['peak_heights']` contains the y-values at detected peaks

**Line 99-101:** Create signal with multiple frequency components to demonstrate FFT

**Line 104:** `np.fft.fft(signal)` computes Fast Fourier Transform:
- Converts time domain → frequency domain
- Returns complex numbers (use `np.abs()` for magnitude)

**Line 105:** `np.fft.fftfreq(n, d)` generates frequency values:
- `n` - number of samples
- `d` - sample spacing (1/sample_rate)

**Line 108-109:** FFT output is symmetric, so we only need positive frequencies (first half)

FFT reveals which frequencies are present in signal - peaks at 5, 10, 20 Hz match our input components.

---

## Key SciPy Concepts Summary

### Module Organization
```python
from scipy import optimize    # Optimization, root finding
from scipy import integrate   # Integration, ODEs
from scipy import interpolate # Interpolation, smoothing
from scipy import stats       # Statistics, distributions
from scipy import signal      # Signal processing
from scipy import linalg      # Linear algebra (advanced)
from scipy import sparse      # Sparse matrices
from scipy import spatial     # Spatial algorithms
```

### Common Workflows

**Curve Fitting:**
1. Define model function
2. Provide initial guess
3. Call `curve_fit()`
4. Extract uncertainties from covariance matrix

**Solving ODEs:**
1. Define ODE as `f(t, y)` returning derivatives
2. Set initial conditions
3. Call `solve_ivp()`
4. Extract solution from result object

**Interpolation:**
1. Create interpolation function with data
2. Call function with new x values
3. Don't extrapolate far beyond data

**Statistical Testing:**
1. Choose appropriate test (t-test, KS test, etc.)
2. Check assumptions (normality, independence)
3. Interpret p-value (< 0.05 typically significant)

### When to Use What

**Optimization Methods:**
- BFGS: General purpose, gradient-based
- Nelder-Mead: Derivative-free, slower
- Powell: Derivative-free, good for noisy functions
- L-BFGS-B: Constrained optimization with bounds

**Interpolation Types:**
- Linear: Fast, continuous but not smooth
- Quadratic: Smooth, can oscillate
- Cubic: Smooth, standard choice
- Splines: Flexible, smoothing option available

**Statistical Tests:**
- T-test: Compare means of two groups
- ANOVA: Compare means of multiple groups
- KS test: Compare distributions
- Chi-square: Categorical data, goodness of fit
- Pearson: Linear correlation
- Spearman: Monotonic correlation

**Filters:**
- Savitzky-Golay: Preserves peaks, good for derivatives
- Moving average: Simple, blurs features
- Median: Removes spikes, preserves edges
- Butterworth: Frequency-selective filtering

## Common Gotchas

1. **Curve fitting convergence**: Provide good initial guesses (`p0`). Bad guesses = failed fits.

2. **ODE function signature**: Must be `f(t, y)` not `f(y, t)`. Order matters!

3. **Interpolation extrapolation**: Don't trust values outside data range. Use `fill_value` parameter to control behavior.

4. **P-value misinterpretation**: p < 0.05 doesn't mean large effect, just statistical significance.

5. **FFT frequency units**: Match your sample rate. If sampling at 100 Hz, frequencies are in Hz.

6. **Integration accuracy**: `quad()` is usually fine, but check error estimate. For oscillatory functions, may need specialized methods.

---

## Practice Exercises

1. Fit a power law model `y = A * x^b` to data points. Extract parameter uncertainties.

2. Solve the damped harmonic oscillator ODE: `d²y/dt² + 2ζω₀ dy/dt + ω₀²y = 0`

3. Interpolate sparse 2D data on a fine grid and create a contour plot.

4. Generate 1000 samples from chi-square distribution, perform KS test against theoretical distribution.

5. Create a noisy signal with 3 frequency components, apply FFT, identify the frequencies.

## Additional Resources

- Official Documentation: https://docs.scipy.org/doc/scipy/
- SciPy Cookbook: https://scipy-cookbook.readthedocs.io/
- Stats Tutorial: https://docs.scipy.org/doc/scipy/tutorial/stats.html
- Optimization Guide: https://docs.scipy.org/doc/scipy/tutorial/optimize.html
