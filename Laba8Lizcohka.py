import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from itertools import cycle

# ------------------------------------------------------------------
# Problem specification (individual variant)
#   y'' + y'/x + 2y = x, on [a,b] = [0.7, 1.0]
#   y(0.7) = 0.5
#   2*y(1) + 3*y'(1) = 1.2
# ------------------------------------------------------------------

# Problem data
a, b = 0.7, 1.0
alpha0, alpha1, A = 1.0, 0.0, 0.5   # y(a)=A/alpha0
beta0, beta1, B = 2.0, 3.0, 1.2     # BC at x=b: beta0*y + beta1*y' = B

def p(x):    return 1.0/x
def q(x):    return 2.0
def f_rhs(x):return x

# ------------------------------------------------------------------
# Thomas algorithm for tridiagonal system
# ------------------------------------------------------------------
def thomas_algorithm(a_diag, b_diag, c_diag, d):
    n = len(d)
    cp = np.zeros(n)
    dp = np.zeros(n)
    cp[0] = c_diag[0] / b_diag[0]
    dp[0] = d[0]       / b_diag[0]
    for i in range(1, n):
        denom   = b_diag[i] - a_diag[i]*cp[i-1]
        cp[i]   = (c_diag[i]/denom) if i<n-1 else 0.0
        dp[i]   = (d[i] - a_diag[i]*dp[i-1]) / denom
    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n-2, -1, -1):
        x[i] = dp[i] - cp[i]*x[i+1]
    return x

# ------------------------------------------------------------------
# Finite difference solver (Level B)
# ------------------------------------------------------------------
def finite_difference(n, bc_order=1):
    h = (b - a) / n
    x = a + np.arange(n+1)*h
    y0 = A/alpha0

    N = n
    a_diag = np.zeros(N)
    b_diag = np.zeros(N)
    c_diag = np.zeros(N)
    d       = np.zeros(N)

    # interior nodes i=1..n-1
    for i in range(1, n):
        xi = x[i]
        Ai = 1/h**2 - p(xi)/(2*h)
        Bi = -2/h**2 + q(xi)
        Ci =  1/h**2 + p(xi)/(2*h)
        Fi = f_rhs(xi)
        idx = i-1
        a_diag[idx], b_diag[idx], c_diag[idx] = Ai, Bi, Ci
        d[idx] = Fi - (Ai*y0 if i==1 else 0.0)

    # boundary at i=n
    if bc_order == 1:
        a_diag[-1] = -beta1/h
        b_diag[-1] =  beta0 + beta1/h
        c_diag[-1] =  0.0
        d[-1]      =  B
    else:
        xi = x[n]
        An = 1/h**2 - p(xi)/(2*h)
        Bn = -2/h**2 + q(xi)
        Cn =  1/h**2 + p(xi)/(2*h)
        Fn = f_rhs(xi)
        a_diag[-1] = An + Cn
        b_diag[-1] = Bn - 2*h*Cn*beta0/beta1
        c_diag[-1] = 0.0
        d[-1]      = Fn - 2*h*Cn*B/beta1

    y_inner = thomas_algorithm(a_diag, b_diag, c_diag, d)
    y = np.zeros(n+1)
    y[0]    = y0
    y[1:]   = y_inner
    return x, y

# ------------------------------------------------------------------
# Shooting method (Level C): RK4 + bisection/Newton
# ------------------------------------------------------------------
def shooting_method(method='bisection', tol=1e-6, max_iter=50):
    y0 = A/alpha0

    def phi(s):
        def sys(x, Y):
            return np.array([Y[1], f_rhs(x) - p(x)*Y[1] - q(x)*Y[0]])
        Y = np.array([y0, s])
        M = 1000
        hM = (b - a)/M
        x_val = a
        for _ in range(M):
            k1 = sys(x_val,       Y)
            k2 = sys(x_val+hM/2,  Y + hM*k1/2)
            k3 = sys(x_val+hM/2,  Y + hM*k2/2)
            k4 = sys(x_val+hM,    Y + hM*k3)
            Y += hM*(k1 + 2*k2 + 2*k3 + k4)/6
            x_val += hM
        yb, ypb = Y
        return beta0*yb + beta1*ypb - B

    if method == 'bisection':
        s1, s2 = -10.0, 10.0
        f1, f2 = phi(s1), phi(s2)
        if f1*f2 > 0:
            raise ValueError("Bisection bracket failure")
        for it in range(1, max_iter+1):
            sm = 0.5*(s1 + s2)
            fm = phi(sm)
            if abs(fm) < tol:
                return sm, it
            if f1*fm < 0:
                s2, f2 = sm, fm
            else:
                s1, f1 = sm, fm
        raise RuntimeError("Bisection did not converge")
    else:
        s = 0.0
        for it in range(1, max_iter+1):
            fs   = phi(s)
            dphi = (phi(s+tol) - phi(s-tol)) / (2*tol)
            if abs(dphi) < 1e-14:
                raise ZeroDivisionError("Zero derivative in Newton")
            s_new = s - fs/dphi
            if abs(s_new - s) < tol:
                return s_new, it
            s = s_new
        raise RuntimeError("Newton did not converge")

# ------------------------------------------------------------------
# Main interface
# ------------------------------------------------------------------
def main():
    print("Выберите метод решения:")
    print("  1 – конечные разности + прогонка (Level B)")
    print("  2 – метод стрельбы (Level C)")
    choice = input("Ваш выбор [1/2]: ") or "1"

    eps_tol       = float(input("Точность ε [1e-5]: ") or "1e-5")
    #max_doublings = int(input("Число удвоений [3]: ") or "3")
    #init_n        = int(input("Начальное n [10]: ") or "10")

    # build array of n values
    ns = [10, 20, 40, 80]
    print(f"\nРассматриваем n = {ns}\n")

    if choice == "1":
        bc_order = int(input("Порядок BC (1 или 2) [2]: ") or "2")
        print("--- Finite Difference (Level B) ---")

        # exact solution via very fine FD
        x_ref, y_ref = finite_difference(512, bc_order)

        summary = []
        plt.figure(figsize=(8,5))
        plt.plot(x_ref, y_ref, 'k-', linewidth=1.5, label='точное решение')
        colors = cycle(plt.cm.Set1.colors)

        for n in ns:
            x1, y1 = finite_difference(n, bc_order)
            x2, y2 = finite_difference(2*n, bc_order)
            err = np.max(np.abs(y2[::2] - y1))

            tbl = [
                ["x_i"]    + [f"{xi:.5f}" for xi in x1],
                ["y_ord1"] + [f"{yi:.8f}" for yi in y1],
                ["y_ord2"] + [f"{yi:.8f}" for yi in y2[::2]],
                ["|Δ|"]    + [f"{abs(y2i-y1i):.3e}"
                              for y2i,y1i in zip(y2[::2], y1)]
            ]
            print(f"\n=== n={n}, BC-order={bc_order}, ε={err:.3e} ===")
            print(tabulate(tbl, tablefmt="grid"))

            summary.append([n, (b-a)/n, err, (b-a)/(2*n), err])
            plt.plot(x1, y1, '--', color=next(colors), label=f"FD n={n}")

        print("\n>>> Итоговая сводная таблица FD:")
        print(tabulate(
            summary,
            headers=["n","h","ε","h/2","ε₂"],
            tablefmt="grid",
            floatfmt=(".0f",".5f",".3e",".5f",".3e")
        ))

        plt.title("Convergence of FD")
        plt.xlabel("x"); plt.ylabel("y")
        plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    else:
        print("--- Shooting Method (Level C) ---")
        # exact via fine FD
        x_ref, y_ref = finite_difference(512, 2)

        for n in ns:
            xs = np.linspace(a, b, n+1)
            sb, itb = shooting_method('bisection', eps_tol)
            sn, itn = shooting_method('newton',   eps_tol)

            # integrate with RK4 using n steps
            def integrate(s):
                Y = np.array([A/alpha0, s])
                sol = [Y.copy()]
                h = (b - a)/n
                x = a
                for _ in range(n):
                    def sys(x, Y):
                        return np.array([Y[1], f_rhs(x) - p(x)*Y[1] - q(x)*Y[0]])
                    k1 = sys(x,        Y)
                    k2 = sys(x+h/2,    Y + h*k1/2)
                    k3 = sys(x+h/2,    Y + h*k2/2)
                    k4 = sys(x+h,      Y + h*k3)
                    Y = Y + h*(k1 + 2*k2 + 2*k3 + k4)/6
                    x += h
                    sol.append(Y.copy())
                return np.array(sol)[:,0]

            yb = integrate(sb)
            yn = integrate(sn)
            err = np.max(np.abs(yb - yn))

            print(f"\n=== n={n}, max|Δ|={err:.3e} ===")
            print("Метод       Error  Итераций   s_calc")
            print(f"{'bisect':>10}  {0 if err<eps_tol else 'Over':>5}   {itb:>5}   {sb:>10.5f}")
            print(f"{'newton':>10}  {0 if err<eps_tol else 'Over':>5}   {itn:>5}   {sn:>10.5f}")

            tbl = [
                ["x_i"]      + [f"{xi:.5f}" for xi in xs],
                ["y_bisect"] + [f"{yi:.8f}" for yi in yb],
                ["y_newton"] + [f"{zi:.8f}" for zi in yn],
                ["|Δ|"]      + [f"{abs(yi-zi):.3e}" for yi,zi in zip(yb,yn)]
            ]
            print("\nТаблица решения (уровень C):")
            print(tabulate(tbl, tablefmt="grid"))

            plt.figure(figsize=(6,4))
            plt.plot(x_ref, y_ref, 'k-', label='точное решение')
            plt.plot(xs,     yb,    'b--', label=f'bisect n={n}')
            plt.plot(xs,     yn,    'r-.', label=f'newton n={n}')
            plt.title(f"Shooting comparison, n={n}, max|Δ|={err:.2e}")
            plt.xlabel("x"); plt.ylabel("y")
            plt.legend(); plt.grid(); plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    main()
