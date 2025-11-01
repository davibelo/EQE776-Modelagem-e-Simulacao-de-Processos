import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

R = 5.0
Pe = 15.0
Da = 1.0
CF = 1.0
QF = 10.0

N = 60
dz = 1.0 / (N - 1)

T_sim = 5.0

def model(t, C_vec):
    C = C_vec.copy()
    
    C_saida = C[-1]
    C_alim = (QF * CF + R * QF * C_saida) / (QF * (1 + R))
    
    dCdt = np.zeros(N)
    
    i = 0
    C_left = C_alim - (1 / Pe) * (C[1] - C_alim) / dz
    dCdt[i] = -(C[i] - C_left) / dz + (1 / Pe) * (C[i+1] - 2*C[i] + C_left) / (dz**2) - Da * C[i]
    
    for i in range(1, N-1):
        dCdt[i] = -(C[i] - C[i-1]) / dz + (1 / Pe) * (C[i+1] - 2*C[i] + C[i-1]) / (dz**2) - Da * C[i]
    
    i = N - 1
    C_right = C[i-1]
    dCdt[i] = -(C[i] - C[i-1]) / dz + (1 / Pe) * (C_right - 2*C[i] + C[i-1]) / (dz**2) - Da * C[i]
    
    return dCdt

C0 = np.zeros(N)

sol = solve_ivp(
    fun=model,
    t_span=(0.0, T_sim),
    y0=C0,
    max_step=0.01,
    dense_output=True
)

t_vals = np.linspace(0.0, T_sim, 300)
C_all = sol.sol(t_vals)

z_vals = np.linspace(0.0, 1.0, N)

idx_025 = int(0.25 / dz)
idx_050 = int(0.50 / dz)
idx_075 = int(0.75 / dz)
idx_100 = N - 1

C_025 = C_all[idx_025, :]
C_050 = C_all[idx_050, :]
C_075 = C_all[idx_075, :]
C_100 = C_all[idx_100, :]

C_alim_vals = np.zeros_like(t_vals)
for j, t in enumerate(t_vals):
    C_saida = C_all[-1, j]
    C_alim_vals[j] = (QF * CF + R * QF * C_saida) / (QF * (1 + R))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

ax1.plot(t_vals, C_alim_vals, label="C_alim", color='red', linewidth=2)
ax1.set_xlabel("Tempo (adimensional)")
ax1.set_ylabel("Concentração C_alim (adimensional)")
ax1.legend()
ax1.grid(True)

ax2.plot(t_vals, C_025, label="z = 0.25", color='blue')
ax2.plot(t_vals, C_050, label="z = 0.50", color='green')
ax2.plot(t_vals, C_075, label="z = 0.75", color='orange')
ax2.plot(t_vals, C_100, label="z = 1.00", color='purple')
ax2.set_xlabel("Tempo (adimensional)")
ax2.set_ylabel("Concentração (adimensional)")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("figuras/questao4_reator_dispersao.png", dpi=300)

with open("figuras/questao4_reator_dispersao.dat", "w", encoding="utf-8") as f:
    f.write("tempo C_alim C_z025 C_z050 C_z075 C_z100\n")
    for t, ca, c1, c2, c3, c4 in zip(t_vals, C_alim_vals, C_025, C_050, C_075, C_100):
        f.write(f"{t:.6f} {ca:.6f} {c1:.6f} {c2:.6f} {c3:.6f} {c4:.6f}\n")

print("C_alim final:", C_alim_vals[-1])
print("C (z=0.25) final:", C_025[-1])
print("C (z=0.50) final:", C_050[-1])
print("C (z=0.75) final:", C_075[-1])
print("C (z=1.00) final:", C_100[-1])
