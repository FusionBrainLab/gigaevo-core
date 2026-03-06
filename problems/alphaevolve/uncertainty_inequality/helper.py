import numpy as np
from scipy.special import hermite


def _build_P_from_hermite_coeffs(hermite_coeffs: np.ndarray, degrees: list[int]) -> np.poly1d:
    max_degree = degrees[-1]
    hermite_polys = [hermite(d) for d in degrees]

    P_poly_coeffs = np.zeros(max_degree + 1)
    for i, c in enumerate(hermite_coeffs):
        poly = hermite_polys[i]
        pad_amount = max_degree - poly.order
        P_poly_coeffs[pad_amount:] += c * poly.coef

    if P_poly_coeffs[0] < 0:
        P_poly_coeffs = -P_poly_coeffs
        hermite_coeffs[:] = -hermite_coeffs
    return np.poly1d(P_poly_coeffs)


def _c_and_rmax_from_hermite_coeffs(hermite_coeffs: np.ndarray, num_hermite_coeffs: int):
    degrees = [4 * k for k in range(num_hermite_coeffs)]
    P = _build_P_from_hermite_coeffs(hermite_coeffs.copy(), degrees)

    Q, R = np.polydiv(P, np.poly1d([1.0, 0.0, 0.0]))
    if np.max(np.abs(R.c)) > 1e-10:
        return None, None

    roots = Q.r
    real_pos = roots[(np.isreal(roots)) & (roots.real > 0)].real
    if real_pos.size == 0:
        return None, None

    r_candidates = np.sort(real_pos)
    r_max = None
    for r in r_candidates:
        eps = 1e-10 * max(1.0, abs(r))
        left = np.polyval(Q, r - eps)
        right = np.polyval(Q, r + eps)
        if left * right < 0:
            r_max = float(r)
    if r_max is None:
        r_max = float(r_candidates[-1])

    c = (r_max**2) / (2 * np.pi)
    return c, r_max


def compute_c_and_rmax(coefficients: np.ndarray):
    coefficients = np.asarray(coefficients, dtype=float)
    num_hermite_coeffs = len(coefficients)
    
    c, rmax = _c_and_rmax_from_hermite_coeffs(coefficients.copy(), num_hermite_coeffs)
    
    if c is None:
        raise ValueError("Failed to compute C - invalid polynomial")
    
    return c, rmax
