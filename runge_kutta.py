import numpy as np

def rk3(A, bvector, yo, interval, N):
    """
    Implements the explicit third-order Runge-Kutta method for solving the ODE system dy/dx = A y + bvector.

    Parameters:
    A (numpy.ndarray): The matrix A in the system dy/dx = A y + bvector. Should be a 2D array of shape (d, d) where d is the dimension.
    bvector (numpy.ndarray): The vector b in the system. Should be a 1D array of shape (d,).
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
    if not isinstance(bvector, np.ndarray) or bvector.shape != (d,):
        raise ValueError("bvector must be a 1D numpy array of shape ({},).".format(d))
    if not isinstance(yo, np.ndarray) or yo.shape != (d,):
        raise ValueError("yo must be a 1D numpy array of shape ({},).".format(d))
    if not isinstance(interval, (list, tuple)) or len(interval) != 2:
        raise ValueError("interval must be a list or tuple of two numbers [a, b].")
    a, b = interval
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or a >= b:
        raise ValueError("interval must be [a, b] with a < b.")
    if not isinstance(N, (int, np.integer)) or N <= 0:
        raise ValueError("N must be a positive integer.")

    # Butcher tableau for explicit RK3
    c = np.array([0, 0.5, 1])
    A_rk = np.array([
        [0, 0, 0],
        [0.5, 0, 0],
        [-1, 2, 0]
    ])
    b_rk = np.array([1/6, 2/3, 1/6])

    h = (b - a) / N
    x = np.linspace(a, b, N+1)
    y = np.zeros((N+1, d))
    y[0] = yo

    for n in range(N):
        k1 = A @ y[n] + bvector
        k2 = A @ (y[n] + h * A_rk[1,0] * k1) + bvector
        k3 = A @ (y[n] + h * (A_rk[2,0] * k1 + A_rk[2,1] * k2)) + bvector
        y[n+1] = y[n] + h * (b_rk[0] * k1 + b_rk[1] * k2 + b_rk[2] * k3)

    return x, y

def dirk3(A, bvector, yo, interval, N):
    """
    Implements the diagonally implicit third-order Runge-Kutta method for solving the ODE system dy/dx = A y + bvector.

    Parameters:
    A (numpy.ndarray): The matrix A in the system dy/dx = A y + bvector. Should be a 2D array of shape (d, d) where d is the dimension.
    bvector (numpy.ndarray): The vector b in the system. Should be a 1D array of shape (d,).
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
    if not isinstance(bvector, np.ndarray) or bvector.shape != (d,):
        raise ValueError("bvector must be a 1D numpy array of shape ({},).".format(d))
    if not isinstance(yo, np.ndarray) or yo.shape != (d,):
        raise ValueError("yo must be a 1D numpy array of shape ({},).".format(d))
    if not isinstance(interval, (list, tuple)) or len(interval) != 2:
        raise ValueError("interval must be a list or tuple of two numbers [a, b].")
    a, b = interval
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or a >= b:
        raise ValueError("interval must be [a, b] with a < b.")
    if not isinstance(N, (int, np.integer)) or N <= 0:
        raise ValueError("N must be a positive integer.")

    # Butcher tableau for DIRK3
    gamma = (3 + np.sqrt(3)) / 6
    c = np.array([gamma, 1, 1])
    A_rk = np.array([
        [gamma, 0, 0],
        [1 - 2*gamma, gamma, 0],
        [0.5 - gamma, 2*gamma - 1, gamma]
    ])
    b_rk = np.array([0, 0.5, 0.5])

    h = (b - a) / N
    x = np.linspace(a, b, N+1)
    y = np.zeros((N+1, d))
    y[0] = yo

    for n in range(N):
        # Stage 1
        rhs1 = A @ y[n] + bvector
        k1 = np.linalg.solve(np.eye(d) - h * gamma * A, rhs1)

        # Stage 2
        rhs2 = A @ (y[n] + h * A_rk[1,0] * k1) + bvector
        k2 = np.linalg.solve(np.eye(d) - h * gamma * A, rhs2)

        # Stage 3
        rhs3 = A @ (y[n] + h * (A_rk[2,0] * k1 + A_rk[2,1] * k2)) + bvector
        k3 = np.linalg.solve(np.eye(d) - h * gamma * A, rhs3)

        y[n+1] = y[n] + h * (b_rk[0] * k1 + b_rk[1] * k2 + b_rk[2] * k3)

    return x, y