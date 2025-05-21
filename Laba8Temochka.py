# main.py ― два способа решения краевой задачи
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from itertools import cycle

# ─── 1. Постановка задачи ──────────────────────────────────────────────
def variant_6():
    p  = lambda x: -0.5
    q  = lambda x:  3.0
    f  = lambda x:  2.0 * x**2
    a, b = 1.0, 2.0
    alpha0, alpha1, A_val = 1.0, 2.0, 0.6
    beta0,  beta1,  B_val = 1.0, 0.0, 0.0

    # точное решение
    A_p, B_p, C_p = 2/3, 2/9, -11/27
    y_p   = lambda x: A_p*x**2 + B_p*x + C_p
    y_p_d = lambda x: 2*A_p*x + B_p
    k = np.sqrt(47.0)/4.0
    r = 0.25
    exp, cos, sin = np.exp, np.cos, np.sin

    ex1, ex2   = exp(r*a), exp(r*b)
    cos1, sin1 = cos(k*a), sin(k*a)
    cos2, sin2 = cos(k*b), sin(k*b)
    M = np.array([
        [ex1*cos1 + 2*ex1*(r*cos1 - k*sin1),
         ex1*sin1 + 2*ex1*(r*sin1 + k*cos1)],
        [ex2*cos2, ex2*sin2]
    ])
    rhs = np.array([
        A_val - y_p(a) - 2*y_p_d(a),
        -y_p(b)
    ])
    C1, C2 = np.linalg.solve(M, rhs)

    def y_exact(x):
        x = np.asarray(x)
        y_h = np.exp(r*x)*(C1*np.cos(k*x) + C2*np.sin(k*x))
        return y_h + y_p(x)

    return p, q, f, a, b, alpha0, alpha1, A_val, beta0, beta1, B_val, y_exact

# ─── 2. Метод конечных разностей + Thomas ─────────────────────────────
def fd_solver(p, q, f, a, b,
              alpha0, alpha1, A_val,
              beta0,  beta1,  B_val,
              n, bc_order):
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    N = n+1

    a_diag = np.zeros(N)
    b_diag = np.zeros(N)
    c_diag = np.zeros(N)
    d_rhs  = np.zeros(N)

    # внутренние узлы
    for i in range(1, n):
        a_diag[i] = 1/h**2 - p(x[i])/(2*h)
        b_diag[i] = -2/h**2 + q(x[i])
        c_diag[i] = 1/h**2 + p(x[i])/(2*h)
        d_rhs[i]  = f(x[i])

    # левая граница
    if bc_order == 1:
        b_diag[0] = alpha0 - alpha1/h
        c_diag[0] = alpha1/h
        d_rhs[0]  = A_val
    else:
        a1 = 1/h**2 - p(x[1])/(2*h)
        b1 = -2/h**2 + q(x[1])
        c1 = 1/h**2 + p(x[1])/(2*h)
        f1 = f(x[1])
        B0 = alpha1*a1 + (2*h*alpha0 - 3*alpha1)*c1
        C0 = alpha1*b1 + 4*alpha1*c1
        D0 = 2*h*A_val*c1 + alpha1*f1
        b_diag[0] = B0
        c_diag[0] = C0
        d_rhs[0]  = D0

    # правая граница
    if bc_order == 1:
        a_diag[n] = -beta1/h
        b_diag[n] =  beta0 + beta1/h
        d_rhs[n]  = B_val
    else:
        ai = 1/h**2 - p(x[n-1])/(2*h)
        bi = -2/h**2 + q(x[n-1])
        ci = 1/h**2 + p(x[n-1])/(2*h)
        fi = f(x[n-1])
        a_diag[n] = -beta1*(bi + 4*ai)
        b_diag[n] = (2*h*beta0 + 3*beta1)*ai - beta1*ci
        c_diag[n] = 0.0
        d_rhs[n]  = 2*h*B_val*ai - beta1*fi

    # Thomas algorithm
    alpha = np.zeros(N)
    beta  = np.zeros(N)
    alpha[0] = -c_diag[0] / b_diag[0]
    beta[0]  =  d_rhs[0]  / b_diag[0]
    for i in range(1, N):
        denom    = b_diag[i] + a_diag[i]*alpha[i-1]
        alpha[i] = -c_diag[i] / denom
        beta[i]  = (d_rhs[i] - a_diag[i]*beta[i-1]) / denom

    y = np.zeros(N)
    y[-1] = beta[-1]
    for i in range(N-2, -1, -1):
        y[i] = alpha[i]*y[i+1] + beta[i]

    return x, y

# ─── 3. Метод стрельбы ────────────────────────────────────────────────
def shooting_solver(p, q, f, a, b,
                    alpha0, alpha1, A_val,
                    beta0,  beta1,  B_val,
                    n, eps=1e-6, max_iter=50):
    h = (b-a)/n
    xs = np.linspace(a, b, n+1)

    def rk4_step(Y, x):
        def dY(x, Y):
            y, z = Y
            return np.array([z, -p(x)*z - q(x)*y + f(x)])
        k1 = dY(x,      Y)
        k2 = dY(x+h/2,  Y+h*k1/2)
        k3 = dY(x+h/2,  Y+h*k2/2)
        k4 = dY(x+h,    Y+h*k3)
        return Y + (h/6)*(k1+2*k2+2*k3+k4)

    def make_phi():
        def φ(s):
            if alpha1 != 0:
                y0 = (A_val - alpha1*s)/alpha0
            else:
                y0 = A_val/alpha0
            Y = np.array([y0, s])
            x = a
            for _ in range(n):
                Y = rk4_step(Y, x)
                x += h
            yb, zb = Y
            return beta0*yb + beta1*zb - B_val
        return φ

    φ = make_phi()
    info = {}
    sols = {}

    # Bisect
    sL, sR = -1.0, 1.0
    while φ(sL)*φ(sR) > 0:
        sL *= 2; sR *= 2
    for it in range(1, max_iter+1):
        sM = 0.5*(sL+sR)
        if φ(sL)*φ(sM) <= 0:
            sR = sM
        else:
            sL = sM
        if abs(sR-sL) < eps:
            break
    code_b = 0 if abs(sR-sL)<eps else 'Over'
    s_b   = 0.5*(sL+sR)
    info['bisect'] = (code_b, it, s_b)

    # Newton
    sN = 0.0
    for itN in range(1, max_iter+1):
        val  = φ(sN)
        dval = (φ(sN+eps)-φ(sN-eps))/(2*eps)
        if abs(dval)<1e-14: break
        sN -= val/dval
        if abs(val)<eps: break
    code_n = 0 if abs(φ(sN))<eps else 'Over'
    info['newton'] = (code_n, itN, sN)

    # Integrate both
    def integrate(s):
        if alpha1 != 0:
            y0 = (A_val - alpha1*s)/alpha0
        else:
            y0 = A_val/alpha0
        Y = np.array([y0, s])
        sol = [Y.copy()]; x=a
        for _ in range(n):
            Y = rk4_step(Y, x); x+=h
            sol.append(Y.copy())
        return np.array(sol)[:,0]

    sols['bisect'] = integrate(s_b)
    sols['newton'] = integrate(sN)
    return xs, sols, info

# ─── 4. Основная функция ────────────────────────────────────────────────
def main():
    p, q, f, a, b, alpha0, alpha1, A_val, beta0, beta1, B_val, y_exact = variant_6()

    print("Выберите метод решения:")
    print("  1 – конечные разности + прогонка")
    print("  2 – метод стрельбы")
    choice = input("Ваш выбор [1/2]: ") or "1"

    eps_tol       = float(input("Точность ε [1e-5]: ") or "1e-5")
    max_doublings = int(input("Максимум удвоений n [8]: ") or "8")

    # Ветвь 1: FD + прогонка
    if choice == "1":
        #bc_order = int(input("Порядок BC (1 или 2) [2]: ") or "2")
        summary = []
        data1 = []   # для 1-го порядка
        data2 = []   # для 2-го порядка

        print("\n--- FD: промежуточные таблицы ---")
        n = 2
        for _ in range(max_doublings):
            h = (b - a) / n
            # 1-й порядок BC
            x1, y1 = fd_solver(p,q,f,a,b,alpha0,alpha1,A_val,
                               beta0,beta1,B_val,n,1)
            err1   = np.max(np.abs(y1 - y_exact(x1)))
            tbl1 = [
                ["x_i"]     + [f"{xx:.5f}" for xx in x1],
                ["y_exact"] + [f"{y_exact(xx):.8f}" for xx in x1],
                ["y_num1"]  + [f"{yy:.8f}" for yy in y1],
                ["|Δ1|"]    + [f"{abs(y_exact(xx)-yy):.3e}"
                                for xx,yy in zip(x1,y1)]
            ]
            print(f"\n=== n={n}, 1-порядок BC ===")
            print(tabulate(tbl1, tablefmt="grid"))
            data1.append((n, x1, y1))

            # 2-й порядок BC
            x2, y2 = fd_solver(p,q,f,a,b,alpha0,alpha1,A_val,
                               beta0,beta1,B_val,n,2)
            err2   = np.max(np.abs(y2 - y_exact(x2)))
            tbl2 = [
                ["x_i"]     + [f"{xx:.5f}" for xx in x2],
                ["y_exact"] + [f"{y_exact(xx):.8f}" for xx in x2],
                ["y_num2"]  + [f"{yy:.8f}" for yy in y2],
                ["|Δ2|"]    + [f"{abs(y_exact(xx)-yy):.3e}"
                                for xx,yy in zip(x2,y2)]
            ]
            print(f"\n=== n={n}, 2-порядок BC ===")
            print(tabulate(tbl2, tablefmt="grid"))
            data2.append((n, x2, y2))

            summary.append([n, h, err1, h, err2])
            if err2 < eps_tol:
                break
            n *= 2

        # итоговая таблица
        print("\n>>> Итоговая сводная таблица FD:")
        print(tabulate(summary,
                       headers=["n","h(1)","ε(1)","h(2)","ε(2)"],
                       tablefmt="grid",
                       floatfmt=(".0f",".5f",".3e",".5f",".3e")))

        # два отдельных графика для порядка BC=1 и BC=2
        xs_plot = np.linspace(a, b, 500)
        ys_exact = y_exact(xs_plot)

        # график для первого порядка BC
        plt.figure(figsize=(8,5))
        plt.plot(xs_plot, ys_exact, 'k-', label='точное')
        for n, x1, y1 in data1:
            plt.plot(x1, y1, '--', label=f"n={n}")
        plt.title("FD (1-й порядок BC)")
        plt.xlabel("x"); plt.ylabel("y")
        plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        # график для второго порядка BC
        plt.figure(figsize=(8,5))
        plt.plot(xs_plot, ys_exact, 'k-', label='точное')
        for n, x2, y2 in data2:
            plt.plot(x2, y2, '--', label=f"n={n}")
        plt.title("FD (2-й порядок BC)")
        plt.xlabel("x"); plt.ylabel("y")
        plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    # Ветвь 2: метод стрельбы
    else:
        print("\n--- Стрельба: промежуточные графики и таблицы (уровень B и C) ---")
        n = 2
        for _ in range(max_doublings):
            xs, sols, info = shooting_solver(
                p,q,f,a,b,alpha0,alpha1,A_val,
                beta0,beta1,B_val,n,
                eps=eps_tol, max_iter=50
            )
            y_b, y_n = sols['bisect'], sols['newton']
            err_b = np.max(np.abs(y_b - y_exact(xs)))
            err_n = np.max(np.abs(y_n - y_exact(xs)))

            # ── Уровень C: параметры нелинейного решения
            print(f"\n=== n={n}, ε*≈max(|y_h-y|)={max(err_b,err_n):.3e} ===")
            print("Метод    Error   Iter   s_calc")
            for method in ("bisect","newton"):
                code, it, s = info[method]
                print(f"{method:>7}  {code:^5}   {it:^4}   {s:>9.5f}")

            # ── Уровень C: табличка xi, yi, Δy_i, zi, Δz_i
            tbl = [["x_i"], ["y_i (bisect)"], ["|Δy_i|"], ["z_i (newton)"], ["|Δz_i|"]]
            for xi, yi, zi in zip(xs, y_b, y_n):
                tbl[0].append(f"{xi:.5f}")
                tbl[1].append(f"{yi:.8f}")
                tbl[2].append(f"{abs(y_exact(xi)-yi):.3e}")
                tbl[3].append(f"{zi:.8f}")
                tbl[4].append(f"{abs(y_exact(xi)-zi):.3e}")

            print("\nТаблица решения (уровень C):")
            print(tabulate(tbl, tablefmt="grid"))

            # отдельный график для этого n
            xs_plot = np.linspace(a, b, 500)
            plt.figure(figsize=(6,4))
            plt.plot(xs_plot, y_exact(xs_plot), 'k-', label='точное')
            plt.plot(xs,       y_b, 'b--', label=f'bisect n={n}')
            plt.plot(xs,       y_n, 'r-.', label=f'newton n={n}')
            plt.title(f"Стрельба, n={n}\nε_b={err_b:.2e}, ε_n={err_n:.2e}")
            plt.xlabel("x"); plt.ylabel("y")
            plt.legend(); plt.grid(); plt.tight_layout()
            plt.show()

            if err_b < eps_tol and err_n < eps_tol:
                break
            n *= 2

if __name__ == "__main__":
    main()
