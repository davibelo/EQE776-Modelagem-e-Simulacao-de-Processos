import numpy as np
import matplotlib.pyplot as plt

# Dados
q = 0.1            # m^3/h
V = 0.1            # m^3
k0 = 9703*3600     # 1/h (9703 s^-1 -> h^-1)
dH = 5960          # kcal/kmol  (usa-se (-ΔHr) > 0)
E = 11843          # kcal/kmol
Cp = 500           # kcal/(m^3.K)
hA = 15            # kcal/(h.K)
R = 1.987          # kcal/(kmol.K)
Tcf = 290          # K
Tf = 300           # K
CAf = 10           # kmol/m^3

# Funções
k  = lambda T: k0*np.exp(-E/(R*T))
CA = lambda T: ((q/V)*CAf) / (k(T) + (q/V))
r  = lambda T: k(T)*CA(T)
Qgen = lambda T: V*r(T)*dH
Qrem = lambda T: q*Cp*(T - Tf) + hA*(T - Tcf)
F = lambda T: Qgen(T) - Qrem(T)

# Busca de raízes (varredura + bisseção)
Tmin, Tmax, ngrid = 250.0, 450.0, 2000
Tgrid = np.linspace(Tmin, Tmax, ngrid)
Fgrid = F(Tgrid)

roots = []
for i in range(len(Tgrid)-1):
    if Fgrid[i]*Fgrid[i+1] < 0:
        a, b = Tgrid[i], Tgrid[i+1]
        fa, fb = Fgrid[i], Fgrid[i+1]
        for _ in range(80):  # bisseção
            m = 0.5*(a+b)
            fm = F(m)
            if fa*fm <= 0:
                b, fb = m, fm
            else:
                a, fa = m, fm
        roots.append(0.5*(a+b))

# Mostrar soluções
print("Estados estacionários (T em K, CA em kmol/m^3, r em kmol/(m^3.h)):")
for Tsol in roots:
    CAsol = CA(Tsol)
    rsol  = r(Tsol)
    print(f"T={Tsol:.3f} K, CA={CAsol:.6f}, r={rsol:.6f}")

# Gráfico: calor gerado x removido
plt.figure(figsize=(8,5))
plt.plot(Tgrid, Qgen(Tgrid), label='Calor gerado = V r (-ΔHr)', c='tab:red')
plt.plot(Tgrid, Qrem(Tgrid), label='Calor removido = q Cp (T-Tf) + hA (T-Tcf)', c='tab:blue')
for Tsol in roots:
    plt.axvline(Tsol, color='gray', ls='--', lw=0.7)
    plt.scatter([Tsol], [Qgen(Tsol)], color='black')
    plt.text(Tsol+5, Qgen(Tsol), f"T={Tsol:.1f}K\nCA={CA(Tsol):.3f}", fontsize=8)
plt.xlabel('T (K)')
plt.ylabel('Taxa de calor (kcal/h)')
plt.title('Estados estacionários: interseções Qgerado x Qremovido')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()