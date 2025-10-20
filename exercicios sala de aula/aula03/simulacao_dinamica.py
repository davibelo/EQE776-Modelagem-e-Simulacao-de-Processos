import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import os

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

FIGURE_PATHS = {
    "temperatura": os.path.join(BASE_DIR, "temperatura_vs_tempo.png"),
    "concentracao": os.path.join(BASE_DIR, "concentracao_vs_tempo.png"),
    "espaco_estados": os.path.join(BASE_DIR, "espaco_de_estados.png"),
}
RESULTS_PATH = os.path.join(BASE_DIR, "resultados_simulacao.txt")

# Parâmetros do reator (iguais aos usados antes)
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

# Cinética
k  = lambda T: k0*np.exp(-E/(R*T))

# Balanços diferenciais
def odes(t, y):
    CA, T = y
    r = k(T) * CA
    dCAdt = q*CAf/V - q*CA/V - r
    dTdt  = q*Tf/V - q*T/V + r*dH/Cp - (hA*(T - Tcf))/(V*Cp)
    return np.array([dCAdt, dTdt])

# Integração numérica com solver pronto

T_end = 10.0  # h
n = 1000
Ttime = np.linspace(0, T_end, n)
T0_list = [300, 320, 340, 360, 380, 400]
CA0 = CAf

sols = []
for T0 in T0_list:
    sol = solve_ivp(
        odes,
        (Ttime[0], Ttime[-1]),
        np.array([CA0, float(T0)]),
        t_eval=Ttime,
    )
    if not sol.success:
        raise RuntimeError(f"Falha na integração para T0={T0}: {sol.message}")
    sols.append((T0, sol.t, sol.y.T))

# Estados estacionários para marcar no plano de fase
CA_ss = lambda T: ((q/V)*CAf) / (k(T) + (q/V))
F = lambda T: V*k(T)*CA_ss(T)*dH - (q*Cp*(T-Tf) + hA*(T-Tcf))

# Encontrar raízes de F(T) (pontos de equilíbrio de temperatura)
Ts = np.linspace(250, 900, 3000)
Fv = F(Ts)
roots = []
from scipy.optimize import brentq
for i in range(len(Ts) - 1):
    if Fv[i] * Fv[i + 1] < 0:
        a, b = Ts[i], Ts[i + 1]
        try:
            root = brentq(F, a, b)
            # evitar raízes duplicadas devido a resoluções numéricas
            if not any(abs(root - r) < 1e-6 for r in roots):
                roots.append(root)
        except ValueError:
            continue
roots = sorted(roots)

# Plot T(t)
all_temperatures = np.concatenate([Y[:, 1] for _, _, Y in sols])
all_concentrations = np.concatenate([Y[:, 0] for _, _, Y in sols])
if roots:
    roots_array = np.array(roots)
    eq_conc = np.array([CA_ss(Tss) for Tss in roots_array])
    all_temperatures = np.concatenate([all_temperatures, roots_array])
    all_concentrations = np.concatenate([all_concentrations, eq_conc])

T_min, T_max = all_temperatures.min(), all_temperatures.max()
CA_min, CA_max = all_concentrations.min(), all_concentrations.max()
T_range = T_max - T_min
CA_range = CA_max - CA_min
T_margin = 0.05 * T_range if T_range > 0 else 5
CA_margin = 0.05 * CA_range if CA_range > 0 else 0.5

fig1, ax1 = plt.subplots(figsize=(9, 4.5))
for T0, t, Y in sols:
    ax1.plot(t, Y[:, 1], label=f'T0={T0} K')
if roots:
    for Tss in roots:
        ax1.axhline(Tss, color='gray', ls='--', lw=0.8)
ax1.set_xlabel('tempo (h)')
ax1.set_ylabel('T (K)')
ax1.set_title('Temperatura vs tempo para diferentes condições iniciais')
ax1.legend(ncol=3)
ax1.grid(True, alpha=0.3)
fig1.tight_layout()
fig1.savefig(FIGURE_PATHS["temperatura"], dpi=300)

# Plot CA(t)
fig2, ax2 = plt.subplots(figsize=(9, 4.5))
for T0, t, Y in sols:
    ax2.plot(t, Y[:, 0], label=f'T0={T0} K')
ax2.set_xlabel('tempo (h)')
ax2.set_ylabel('CA (kmol/m^3)')
ax2.set_title('Concentração de A vs tempo')
ax2.legend(ncol=3)
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig(FIGURE_PATHS["concentracao"], dpi=300)

# Plano de espaço de estados (CA x T)
fig3, ax3 = plt.subplots(figsize=(7.5, 6))
for T0, t, Y in sols:
    ax3.plot(Y[:, 1], Y[:, 0], label=f'T0={T0} K')
# marcar equilíbrios (somente se existirem)
if roots:
    for Tss in roots:
        CA_eq = CA_ss(Tss)
        ax3.scatter([Tss], [CA_eq], color='black')
        ax3.text(
            Tss + 0.02 * max(T_range, 1),
            CA_eq + 0.02 * max(CA_range, 0.1),
            f'T={Tss:.1f}',
            fontsize=8
        )
ax3.set_xlabel('T (K)')
ax3.set_ylabel('CA (kmol/m^3)')
ax3.set_title('Diagrama de espaço de estados (CA x T)')
ax3.legend(ncol=2)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(T_min - T_margin, T_max + T_margin)
ax3.set_ylim(CA_min - CA_margin, CA_max + CA_margin)
fig3.tight_layout()
fig3.savefig(FIGURE_PATHS["espaco_estados"], dpi=300)

plt.show()

# Mostrar valores finais de cada trajetória e salvar em arquivo
result_lines = []
for T0, t, Y in sols:
    line = f'T0={T0} K -> T(10h)={Y[-1,1]:.3f} K, CA(10h)={Y[-1,0]:.4f} kmol/m^3'
    print(line)
    result_lines.append(line)
if roots:
    eq_line = 'Equilíbrios T (K): ' + str([round(x, 3) for x in roots])
else:
    eq_line = 'Equilíbrios T (K): []'
print(eq_line)
result_lines.append(eq_line)
with open(RESULTS_PATH, 'w', encoding='utf-8') as results_file:
    results_file.write('\n'.join(result_lines))
for T0, t, Y in sols:
    print(f'T0={T0} K -> T(10h)={Y[-1,1]:.3f} K, CA(10h)={Y[-1,0]:.4f} kmol/m^3')
print('Equilíbrios T (K):', [round(x,3) for x in roots])