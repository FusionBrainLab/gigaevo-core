import numpy as np
import sympy as sp

x = sp.symbols("x")


def _hermite_4k_polys(m: int):
    degrees = [4 * k for k in range(m)]
    Hs = [sp.polys.orthopolys.hermite_poly(n=d, x=x, polys=False) for d in degrees]
    return Hs, degrees


def _construct_P_with_forced_zero(coeffs: np.ndarray) -> sp.Expr:
    m = len(coeffs)
    Hs, _ = _hermite_4k_polys(m + 1)
    rc = [sp.Rational(c) for c in coeffs]

    partial = sum(rc[i] * Hs[i] for i in range(m))
    a = Hs[m].subs(x, 0)
    b = -partial.subs(x, 0)
    c_last = sp.Rational(b) / sp.Rational(a)

    P = partial + c_last * Hs[m]
    if sp.limit(P, x, sp.oo) < 0:
        P = -P

    return sp.simplify(P)


def _largest_positive_root_of_P_over_x2(P: sp.Expr) -> float:
    gq = sp.exquo(P, x**2)
    roots = sp.real_roots(gq, x)
    if not roots:
        raise ValueError("No real roots for P(x)/x^2.")

    best = None
    for r in roots:
        r_approx = r.eval_rational(n=200)
        eps = sp.Rational(1, 10**198)
        left = gq.subs(x, r_approx - eps)
        right = gq.subs(x, r_approx + eps)
        if (left > 0 and right < 0) or (left < 0 and right > 0):
            if best is None or r_approx > best:
                best = r_approx

    if best is None:
        raise ValueError("No root with a verified sign change for P(x)/x^2.")
    return float(best)


def compute_polynomial(x_val, coefficients):
    """Compute polynomial value at x_val using sympy (for validation)."""
    coefficients = np.asarray(coefficients, dtype=float)
    P = _construct_P_with_forced_zero(coefficients)
    return float(P.subs(x, x_val))


def compute_c(coefficients):
    """Compute constant C using sympy (for validation/verification)."""
    coefficients = np.asarray(coefficients, dtype=float)
    
    P = _construct_P_with_forced_zero(coefficients)
    assert P.subs(x, 0) == 0, "P(0) != 0 after forcing."
    assert sp.limit(P, x, sp.oo) > 0, "Limit at +inf is not positive."

    rmax = _largest_positive_root_of_P_over_x2(P)
    c = (rmax**2) / (2.0 * np.pi)
    
    return c


def validate(coefficients):
    coefficients = np.asarray(coefficients, dtype=float)
    
    if coefficients.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {coefficients.shape}")
    
    if coefficients.size == 0:
        raise ValueError("Array cannot be empty")
    
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("Some coefficients are NaN or infinite")
    
    p_at_zero = compute_polynomial(0.0, coefficients)
    if abs(p_at_zero) > 1e-9:
        raise ValueError(
            f"Constraint P(0) = 0 violated. P(0) = {p_at_zero:.2e}"
        )

    eps = 1e-4
    f_near_zero = compute_polynomial(eps, coefficients) * np.exp(-np.pi * eps**2)
    if f_near_zero >= 0:
        raise ValueError(
            f"Sign condition f(0) < 0 violated. f({eps}) = {f_near_zero:.2e}"
        )

    large_x = 5.0
    p_large = compute_polynomial(large_x, coefficients)
    if p_large < 0:
        raise ValueError(
            f"Positivity condition violated. P({large_x}) = {p_large:.2e}"
        )

    try:
        c = compute_c(coefficients)
    except Exception as e:
        raise ValueError(f"Error computing C: {e}")
    
    if not np.isfinite(c) or c <= 0:
        raise ValueError(f"Invalid C value: {c}")
    
    return {"fitness": c, "is_valid": 1}
