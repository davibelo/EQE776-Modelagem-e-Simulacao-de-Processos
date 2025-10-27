import numpy as np
from math import pi
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

Q0 = 20.0  # m^3/h
D1 = 4.0   # m
D2 = 3.0   # m
k1 = 14.0  # m^{2.5}/h
k2 = 12.0  # m^{2.5}/h
h1_0 = 3.0  # m
h2_0 = 2.0  # m

A1 = pi * (D1 ** 2) / 4.0
A2 = pi * (D2 ** 2) / 4.0

Q0 /= 3600.0
k1 /= 3600.0
k2 /= 3600.0

T_sim = 20.0
Te = T_sim * 3600.0

def model(t, y):
    h1, h2 = y
    q1 = k1 * np.sqrt(max(h1, 0.0))
    q2 = k2 * np.sqrt(max(h2, 0.0))
    dh1dt = (Q0 - q1) / A1
    dh2dt = (q1 - q2) / A2
    return [dh1dt, dh2dt]

sol = solve_ivp(model, (0.0, Te), [h1_0, h2_0], max_step=10.0, dense_output=True)

t_hours = np.linspace(0.0, T_sim, 1000)
h1 = sol.sol(t_hours * 3600.0)[0]
h2 = sol.sol(t_hours * 3600.0)[1]

plt.figure(figsize=(6, 4))
plt.plot(t_hours, h1, label="h1 (m)")
plt.plot(t_hours, h2, label="h2 (m)")
plt.xlabel("Tempo (h)")
plt.ylabel("Nível (m)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figuras/questao1_niveis.png", dpi=300)

with open("figuras/questao1_niveis.dat", "w", encoding="utf-8") as f:
    f.write("tempo_h h1_m h2_m\n")
    for t, hv1, hv2 in zip(t_hours, h1, h2):
        f.write(f"{t:.6f} {hv1:.6f} {hv2:.6f}\n")

print("h1 final:", h1[-1])
print("h2 final:", h2[-1])
