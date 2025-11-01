import numpy as np
import matplotlib.pyplot as plt
from runge_kutta import rk3, dirk3

# Test Case 1: Moderately Stiff System
print("Test Case 1: Moderately Stiff System")

a1 = 1000
a2 = 1
A1 = np.array([[-a1, 0], [a1, -a2]])
bvector1 = lambda x: np.zeros(2)
yo1 = np.array([1.0, 0.0])
interval1 = [0, 0.1]

# Exact solution
def exact_y1(x):
    x = np.asarray(x)
    e1 = np.exp(-a1 * x)
    e2 = np.exp(-a2 * x)
    y1 = e1
    y2 = (a1 / (a1 - a2)) * (e2 - e1)
    return np.column_stack([y1, y2])

k_values = np.arange(1, 11)
N_values = 40 * k_values
h_values = 0.1 / N_values

errors_rk3 = []
errors_dirk3 = []

for i, N in enumerate(N_values):
    h = h_values[i]
    x_rk3, y_rk3 = rk3(A1, bvector1, yo1, interval1, N)
    x_dirk3, y_dirk3 = dirk3(A1, bvector1, yo1, interval1, N)
    
    # Get the exact solution at all x points
    y_exact_all = exact_y1(x_rk3)
    
    # Calculate relative error for y2 at all points (excluding j=0)
    relative_err_rk3 = np.abs((y_rk3[1:, 1] - y_exact_all[1:, 1]) / y_exact_all[1:, 1])
    relative_err_dirk3 = np.abs((y_dirk3[1:, 1] - y_exact_all[1:, 1]) / y_exact_all[1:, 1])
    
    # Calculate the 1-norm (sum * h)
    error_rk3 = h * np.sum(relative_err_rk3)
    error_dirk3 = h * np.sum(relative_err_dirk3)
    
    errors_rk3.append(error_rk3)
    errors_dirk3.append(error_dirk3)

errors_rk3 = np.array(errors_rk3)
errors_dirk3 = np.array(errors_dirk3)

# Plot error vs h
plt.figure(figsize=(10, 6))
plt.loglog(h_values, errors_rk3, 'o-', label='RK3')
plt.loglog(h_values, errors_dirk3, 's-', label='DIRK3')
plt.xlabel('Step size h')
plt.ylabel('Relative error for y2')
plt.title('Error vs Step Size for Test Case 1')
plt.legend()
plt.grid(True)
plt.savefig('test_case_1_error_vs_h.png')
plt.show()

# Fit curve for RK3
coeffs_rk3 = np.polyfit(np.log(h_values), np.log(errors_rk3), 1)
slope_rk3 = coeffs_rk3[0]
print(f"RK3 slope: {slope_rk3}")

# Fit for DIRK3
coeffs_dirk3 = np.polyfit(np.log(h_values), np.log(errors_dirk3), 1)
slope_dirk3 = coeffs_dirk3[0]
print(f"DIRK3 slope: {slope_dirk3}")

# Plot numerical solution at N=400
N_high = 400
x_rk3_high, y_rk3_high = rk3(A1, bvector1, yo1, interval1, N_high)
x_dirk3_high, y_dirk3_high = dirk3(A1, bvector1, yo1, interval1, N_high)
x_exact = np.linspace(0, 0.1, 1000)
y_exact_plot = exact_y1(x_exact)

plt.figure(figsize=(10, 6))
plt.plot(x_exact, y_exact_plot[:, 0], 'k-', label='Exact y1')
plt.plot(x_exact, y_exact_plot[:, 1], 'k--', label='Exact y2')
plt.plot(x_rk3_high, y_rk3_high[:, 0], 'b-o', label='RK3 y1', markersize=2)
plt.plot(x_rk3_high, y_rk3_high[:, 1], 'b-s', label='RK3 y2', markersize=2)
plt.plot(x_dirk3_high, y_dirk3_high[:, 0], 'r-o', label='DIRK3 y1', markersize=2)
plt.plot(x_dirk3_high, y_dirk3_high[:, 1], 'r-s', label='DIRK3 y2', markersize=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Numerical Solutions vs Exact for Test Case 1 (N=400)')
plt.legend()
plt.grid(True)
plt.savefig('test_case_1_solutions.png')
plt.show()

# Test Case 2: Stiff System
print("Test Case 2: Stiff System")

# A from Equation 106
A2 = np.array([[-1, 0, 0], [-99, -100, 0], [-10098, 9900, -10000]])
# b(x) from Equation 108
def bvector2(x):
    return np.array([np.cos(10*x) - 10*np.sin(10*x), 199*np.cos(10*x) - 10*np.sin(10*x), 208*np.cos(10*x) + 10000*np.sin(10*x)])
# yo from Equation 109
yo2 = np.array([0.0, 1.0, 0.0])
interval2 = [0, 1]

# Exact solution from Equation 110
def exact_y2(x):
    x = np.asarray(x)
    cos10x = np.cos(10*x)
    sin10x = np.sin(10*x)
    e_x = np.exp(-x)
    e_100x = np.exp(-100*x)
    e_10000x = np.exp(-10000*x)
    y1 = cos10x - e_x
    y2 = cos10x + e_x - e_100x
    y3 = sin10x + 2*e_x - e_100x - e_10000x
    return np.column_stack([y1, y2, y3])

k_values2 = np.arange(4, 17)
N_values2 = 200 * k_values2
h_values2 = 1.0 / N_values2

errors_dirk3_2 = []

for N in N_values2:
    x_dirk3_2, y_dirk3_2 = dirk3(A2, bvector2, yo2, interval2, N)
    
    # Get the exact solution at all x points
    y_exact_all_2 = exact_y2(x_dirk3_2)
    
    # Calculate relative error for y3 at all points (excluding j=0)
    relative_err_dirk3_2 = np.abs((y_dirk3_2[1:, 2] - y_exact_all_2[1:, 2]) / y_exact_all_2[1:, 2])
    
    # Calculate the 1-norm (sum * h)
    h2 = 1.0 / N
    error_dirk3_2 = h2 * np.sum(relative_err_dirk3_2)
    
    errors_dirk3_2.append(error_dirk3_2)

errors_dirk3_2 = np.array(errors_dirk3_2)

# Plot error vs h for DIRK3
plt.figure(figsize=(10, 6))
plt.loglog(h_values2, errors_dirk3_2, 's-', label='DIRK3')
plt.xlabel('Step size h')
plt.ylabel('Relative error for y3')
plt.title('Error vs Step Size for DIRK3 in Test Case 2')
plt.legend()
plt.grid(True)
plt.savefig('test_case_2_error_vs_h.png')
plt.show()

# Fit curve
coeffs_dirk3_2 = np.polyfit(np.log(h_values2), np.log(errors_dirk3_2), 1)
slope_dirk3_2 = coeffs_dirk3_2[0]
print(f"DIRK3 slope for Test Case 2: {slope_dirk3_2}")

# Plot numerical solutions at highest N
N_high2 = 3200
x_rk3_high2, y_rk3_high2 = rk3(A2, bvector2, yo2, interval2, N_high2)
x_dirk3_high2, y_dirk3_high2 = dirk3(A2, bvector2, yo2, interval2, N_high2)
x_exact2 = np.linspace(0, 1, 1000)
y_exact_plot2 = exact_y2(x_exact2)

plt.figure(figsize=(15, 10))

for i in range(3):
    plt.subplot(3, 1, i+1)
    plt.plot(x_exact2, y_exact_plot2[:, i], 'k-', label=f'Exact y{i+1}')
    plt.plot(x_rk3_high2, y_rk3_high2[:, i], 'b-o', label=f'RK3 y{i+1}', markersize=1)
    plt.plot(x_dirk3_high2, y_dirk3_high2[:, i], 'r-s', label=f'DIRK3 y{i+1}', markersize=1)
    plt.xlabel('x')
    plt.ylabel(f'y{i+1}')
    plt.title(f'Component y{i+1} for Test Case 2 (N={N_high2})')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.savefig('test_case_2_solutions.png')
plt.show()