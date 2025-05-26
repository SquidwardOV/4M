import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# ------------------------------------------------------------------
# Задача: y'' + y'/x + 2y = x,  на [a,b]=[0.7,1.0]
#   y(0.7)=0.5,
#   2*y(1)+3*y'(1)=1.2
# ------------------------------------------------------------------

a, b = 0.7, 1.0
alpha0, A = 1.0, 0.5    # y(a)=A/alpha0
beta0, beta1, B = 2.0, 3.0, 1.2

def p(x):     return 1.0/x
def q(x):     return 2.0
def f_rhs(x): return x

# Thomas для прогонки
def thomas(a_diag, b_diag, c_diag, d):
    n = len(d)
    cp = np.zeros(n); dp = np.zeros(n)
    cp[0] = c_diag[0]/b_diag[0]
    dp[0] = d[0]/b_diag[0]
    for i in range(1,n):
        denom = b_diag[i] - a_diag[i]*cp[i-1]
        cp[i]   = (c_diag[i]/denom) if i<n-1 else 0.0
        dp[i]   = (d[i] - a_diag[i]*dp[i-1]) / denom
    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n-2,-1,-1):
        x[i] = dp[i] - cp[i]*x[i+1]
    return x

# Finite Difference для Level B
def finite_difference(n, bc_order=1):
    h = (b-a)/n
    x = a + np.arange(n+1)*h
    y0 = A/alpha0
    N = n
    a_diag = np.zeros(N); b_diag = np.zeros(N)
    c_diag = np.zeros(N); d = np.zeros(N)

    for i in range(1,n):
        xi = x[i]
        Ai =  1/h**2 - p(xi)/(2*h)
        Bi = -2/h**2 + q(xi)
        Ci =  1/h**2 + p(xi)/(2*h)
        Fi = f_rhs(xi)
        idx = i-1
        a_diag[idx], b_diag[idx], c_diag[idx] = Ai, Bi, Ci
        d[idx] = Fi - (Ai*y0 if i==1 else 0.0)

    # BC на правом конце
    if bc_order==1:
        a_diag[-1] = -beta1/h
        b_diag[-1] =  beta0 + beta1/h
        c_diag[-1] = 0.0
        d[-1]      = B
    else:
        xi = x[n]
        An =  1/h**2 - p(xi)/(2*h)
        Bn = -2/h**2 + q(xi)
        Cn =  1/h**2 + p(xi)/(2*h)
        Fn = f_rhs(xi)
        a_diag[-1] = An + Cn
        b_diag[-1] = Bn - 2*h*Cn*beta0/beta1
        c_diag[-1] = 0.0
        d[-1]      = Fn - 2*h*Cn*B/beta1

    y_inner = thomas(a_diag, b_diag, c_diag, d)
    y = np.zeros(n+1)
    y[0] = y0
    y[1:] = y_inner
    return x, y

# Shooting Method: RK4 + bisection / Newton
def shooting(method='bisection', tol=1e-6, max_iter=50):
    y0 = A/alpha0

    def phi(s):
        def sys(x, Y):
            return np.array([
                Y[1],
                f_rhs(x) - p(x)*Y[1] - q(x)*Y[0]
            ])
        Y = np.array([y0, s])
        M = 1000; hM = (b-a)/M; xv = a
        for _ in range(M):
            k1 = sys(xv,        Y)
            k2 = sys(xv+hM/2,   Y + hM*k1/2)
            k3 = sys(xv+hM/2,   Y + hM*k2/2)
            k4 = sys(xv+hM,     Y + hM*k3)
            Y += hM*(k1+2*k2+2*k3+k4)/6
            xv += hM
        yb, zb = Y
        return beta0*yb + beta1*zb - B

    if method=='bisection':
        s1,s2 = -10.0, 10.0
        f1,f2 = phi(s1), phi(s2)
        if f1*f2>0: raise ValueError("Bracket failure")
        for it in range(1,max_iter+1):
            sm = 0.5*(s1+s2); fm = phi(sm)
            if abs(fm)<tol: return sm, it
            if f1*fm<0:
                s2,f2 = sm,fm
            else:
                s1,f1 = sm,fm
        raise RuntimeError("Bisection no converge")
    else:
        s = 0.0
        for it in range(1,max_iter+1):
            fs   = phi(s)
            dphi = (phi(s+tol)-phi(s-tol))/(2*tol)
            if abs(dphi)<1e-14: raise ZeroDivisionError("Zero deriv")
            s_new = s - fs/dphi
            if abs(s_new-s)<tol: return s_new, it
            s = s_new
        raise RuntimeError("Newton no converge")

# Целиком интегрируем систему RK4 и возвращаем оба компонента
def integrate_full(s, n):
    Y = np.array([A/alpha0, s])
    sol = [Y.copy()]
    h = (b-a)/n; xv = a
    for _ in range(n):
        def sys(x, Y):
            return np.array([Y[1], f_rhs(x)-p(x)*Y[1]-q(x)*Y[0]])
        k1 = sys(xv,        Y)
        k2 = sys(xv+h/2,    Y+h*k1/2)
        k3 = sys(xv+h/2,    Y+h*k2/2)
        k4 = sys(xv+h,      Y+h*k3)
        Y += h*(k1+2*k2+2*k3+k4)/6
        xv += h
        sol.append(Y.copy())
    return np.array(sol)  # shape (n+1, 2)

# ------------------------- MAIN -------------------------
def main():
    print("Выберите метод решения:")
    print(" 1 – FD + прогонка (Level B)")
    print(" 2 – стрельба (Level C)")
    choice = input("Ваш выбор [1/2]: ") or "1"
    eps = float(input("Точность ε [1e-5]: ") or "1e-5")
    ns = [10,20,40,80]

    if choice=="1":
        bc = int(input("Порядок BC (1 или 2) [2]: ") or "2")
        print("--- Finite Difference ---")
        x_ref, y_ref = finite_difference(512, bc)
        plt.figure(figsize=(7,4))
        plt.plot(x_ref, y_ref, 'k-', label='точное')
        for n in ns:
            x1,y1 = finite_difference(n, bc)
            plt.plot(x1,y1,'--',label=f'n={n}')
        plt.legend(); plt.grid(); plt.show()
        return

    # 1) эталон по y
    x_ref, y_ref = finite_difference(512, 2)
    # 2) эталон по z = y'
    z_ref = np.gradient(y_ref, x_ref)

    for n in ns:
        # найдём s_b и s_n
        sb, itb = shooting('bisection', eps)
        sn, itn = shooting('newton',   eps)

        # интегрируем обе системы
        sol_b = integrate_full(sb, n)   # shape (n+1,2)
        sol_n = integrate_full(sn, n)
        xs    = a + np.arange(n+1)*(b-a)/n

        yb = sol_b[:,0]; zb = sol_b[:,1]
        yn = sol_n[:,0]; zn = sol_n[:,1]

        # ошибки
        y_ref_i   = np.interp(xs,    x_ref, y_ref)
        z_ref_i   = np.interp(xs,    x_ref, z_ref)
        err_yb    = np.abs(yb - y_ref_i)
        err_zb    = np.abs(zb - z_ref_i)
        err_yn    = np.abs(yn - y_ref_i)
        err_zn    = np.abs(zn - z_ref_i)

        print(f"\n=== n={n} ===")
        print("Метод       Итераций      s")
        print(f"{'bisect':>8}   {itb:>5}   {sb:>10.5f}")
        print(f"{'newton':>8}   {itn:>5}   {sn:>10.5f}\n")

        # Таблица BISSECTION
        tbl_b = [
            ["x_i"]     + [f"{x:.5g}"   for x in xs],
            ["y_i"]     + [f"{y:.5g}"   for y in yb],
            ["|Δy_i|"]  + [f"{e:.5g}"   for e in err_yb],
            ["z_i"]     + [f"{z:.5g}"   for z in zb],
            ["|Δz_i|"]  + [f"{e:.5g}"   for e in err_zb],
        ]
        print("---- BISSECTION ----")
        print(tabulate(tbl_b, tablefmt="grid"))

        # Таблица NEWTON
        tbl_n = [
            ["x_i"]     + [f"{x:.5g}"   for x in xs],
            ["y_i"]     + [f"{y:.5g}"   for y in yn],
            ["|Δy_i|"]  + [f"{e:.5g}"   for e in err_yn],
            ["z_i"]     + [f"{z:.5g}"   for z in zn],
            ["|Δz_i|"]  + [f"{e:.5g}"   for e in err_zn],
        ]
        print("---- NEWTON ----\n")
        print(tabulate(tbl_n, tablefmt="grid"))

        # и график для наглядности
        plt.figure(figsize=(6,4))
        plt.plot(x_ref, y_ref, 'k-', label='точное y')
        plt.plot(xs, yb, 'b--', label='y bisect')
        plt.plot(xs, yn, 'r-.', label='y newton')
        plt.legend(); plt.grid(); plt.title(f"Shooting, n={n}")
        plt.show()

if __name__=="__main__":
    main()
