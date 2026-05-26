import sympy as sp

x = sp.symbols('x')


def horner(expr):
    return sp.horner(sp.Poly(expr, x)) if getattr(expr, "free_symbols", None) else expr


def cheby_sum_code(n, start=0, func_name="sum_n_cheby"):
    """
    Print each individual Chebyshev polynomial T_k(x) (Horner form),
    then the explicit, non-recursive summed function with S, S', S''.
    """
    lines = []
    # individual polynomials
    for k in range(start, n):
        Tk = sp.expand(sp.chebyshevt(k, x))
        lines.append(f"# T_{k}(x) = {horner(Tk)}")
    lines.append("")

    # the sum and its derivatives
    S  = sp.expand(sum(sp.chebyshevt(k, x) for k in range(start, n)))
    d1 = sp.expand(sp.diff(S, x))
    d2 = sp.expand(sp.diff(S, x, 2))

    lines.append(f"def {func_name}(x):")
    lines.append(f"    # S(x) = sum_{{k={start}}}^{{{n - 1}}} T_k(x)")
    lines.append(f"    val = {horner(S)}")
    lines.append(f"    d1  = {horner(d1)}")
    lines.append(f"    d2  = {horner(d2)}")
    lines.append(f"    return val, d1, d2")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    for n in range(0, 21):  # 0 through 20 inclusive
        Tn = sp.expand(sp.chebyshevt(n, x))                            # the nth Chebyshev
        S  = sp.expand(sum(sp.chebyshevt(k, x) for k in range(n + 1)))  # cumulative sum T_0..T_n
        d1 = sp.expand(sp.diff(S, x))                                   # d/dx of the sum
        d2 = sp.expand(sp.diff(S, x, 2))                               # d2/dx2 of the sum
        print(f"T_{n}(x)      = {horner(Tn)}")
        print(f"sum_0..{n}(x) = {horner(S)}")
        print(f"d1_0..{n}(x)  = {horner(d1)}")
        print(f"d2_0..{n}(x)  = {horner(d2)}")
        print()
