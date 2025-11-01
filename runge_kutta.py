import numpy as np

def rk3(A, bvector, yo, interval, N):
    """
    Implements the explicit third-order Runge-Kutta method as specified in Equation 8.

    Parameters:
    A (numpy.ndarray): The matrix A in the system dy/dx = A y + b(x). Should be a 2D array of shape (d, d) where d is the dimension.
    bvector (callable): The function b(x) in the system. Should be a function that takes a float x and returns a 1D numpy array of shape (d,).
    yo (numpy.ndarray): Initial condition y(0). Should be a 1D array of shape (d,).
    interval (list or tuple): The interval [a, b] over which to solve.
    N (int): Number of steps.

    Returns:
    x (numpy.ndarray): Array of x values from a to b with N+1 points.
    y (numpy.ndarray): Array of y values at each x, shape (N+1, d).
    """
    # Input validation
    if not isinstance(A, np.ndarray) or A.ndim != 2:
        raise ValueError("A must be a 2D numpy array.")
    d = A.shape[0]
    if A.shape[1] != d:
        raise ValueError("A must be square.")
    if not callable(bvector):
        raise ValueError("bvector must be a callable function that takes x and returns a 1D numpy array of shape ({},).".format(d))
    if not isinstance(yo, np.ndarray) or yo.shape != (d,):
        raise ValueError("yo must be a 1D numpy array of shape ({},).".format(d))
    if not isinstance(interval, (list, tuple)) or len(interval) != 2:
        raise ValueError("interval must be a list or tuple of two numbers [a, b].")
    a, b = interval
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or a >= b:
        raise ValueError("interval must be [a, b] with a < b.")
    if not isinstance(N, (int, np.integer)) or N <= 0:
        raise ValueError("N must be a positive integer.")

    h = (b - a) / N
    x = np.linspace(a, b, N+1)
    y = np.zeros((N+1, d))
    y[0] = yo

    for n in range(N):
        x_n = x[n]
        y_n = y[n]
        # y^{(1)} = y_n + h [A y_n + b(x_n)]
        y1 = y_n + h * (A @ y_n + bvector(x_n))
        # y^{(2)} = (3/4) y_n + (1/4) y^{(1)} + (1/4) h [A y^{(1)} + b(x_n + h)]
        y2 = (3/4) * y_n + (1/4) * y1 + (1/4) * h * (A @ y1 + bvector(x_n + h))
        # y_{n+1} = (1/3) y_n + (2/3) y^{(2)} + (2/3) h [A y^{(2)} + b(x_n + h)]
        y[n+1] = (1/3) * y_n + (2/3) * y2 + (2/3) * h * (A @ y2 + bvector(x_n + h))

    return x, y

def dirk3(A, bvector, yo, interval, N):
    """
    Implements the diagonally implicit third-order Runge-Kutta method as specified in Equations 9 and 10.

    Parameters:
    A (numpy.ndarray): The matrix A in the system dy/dx = A y + b(x). Should be a 2D array of shape (d, d) where d is the dimension.
    bvector (callable): The function b(x) in the system. Should be a function that takes a float x and returns a 1D numpy array of shape (d,).
    yo (numpy.ndarray): Initial condition y(0). Should be a 1D array of shape (d,).
    interval (list or tuple): The interval [a, b] over which to solve.
    N (int): Number of steps.

    Returns:
    x (numpy.ndarray): Array of x values from a to b with N+1 points.
    y (numpy.ndarray): Array of y values at each x, shape (N+1, d).
    """
    # Input validation (same as rk3)
    if not isinstance(A, np.ndarray) or A.ndim != 2:
        raise ValueError("A must be a 2D numpy array.")
    d = A.shape[0]
    if A.shape[1] != d:
        raise ValueError("A must be square.")
    if not callable(bvector):
        raise ValueError("bvector must be a callable function that takes x and returns a 1D numpy array of shape ({},).".format(d))
    if not isinstance(yo, np.ndarray) or yo.shape != (d,):
        raise ValueError("yo must be a 1D numpy array of shape ({},).".format(d))
    if not isinstance(interval, (list, tuple)) or len(interval) != 2:
        raise ValueError("interval must be a list or tuple of two numbers [a, b].")
    a, b = interval
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or a >= b:
        raise ValueError("interval must be [a, b] with a < b.")
    if not isinstance(N, (int, np.integer)) or N <= 0:
        raise ValueError("N must be a positive integer.")

    # Coefficients from Equation 10
    mu = 0.5 * (1 - 1 / np.sqrt(3))
    nu = 0.5 * (np.sqrt(3) - 1)
    gamma = 3 / (2 * (3 + np.sqrt(3)))
    lambda_ = 3 * (1 + np.sqrt(3)) / (2 * (3 + np.sqrt(3)))

    h = (b - a) / N
    x = np.linspace(a, b, N+1)
    y = np.zeros((N+1, d))
    y[0] = yo

    M = np.eye(d) - h * mu * A

    for n in range(N):
        x_n = x[n]
        y_n = y[n]
        # Solve for y^{(1)}: [I - h mu A] y^{(1)} = y_n + h mu b(x_n + h mu)
        RHS1 = y_n + h * mu * bvector(x_n + h * mu)
        y1 = np.linalg.solve(M, RHS1)
        # Solve for y^{(2)}: [I - h mu A] y^{(2)} = y^{(1)} + h nu [A y^{(1)} + b(x_n + h mu)] + h mu b(x_n + h nu + 2 h mu)
        RHS2 = y1 + h * nu * (A @ y1 + bvector(x_n + h * mu)) + h * mu * bvector(x_n + h * nu + 2 * h * mu)
        y2 = np.linalg.solve(M, RHS2)
        # y_{n+1} = (1 - lambda) y_n + lambda y^{(2)} + h gamma [A y^{(2)} + b(x_n + h nu + 2 h mu)]
        y[n+1] = (1 - lambda_) * y_n + lambda_ * y2 + h * gamma * (A @ y2 + bvector(x_n + h * nu + 2 * h * mu))

    return x, y