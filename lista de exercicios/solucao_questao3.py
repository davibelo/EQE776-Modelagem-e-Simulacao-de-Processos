import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

CA0 = 0.5
CB0 = 0.8
CC0 = 0.0
CD0 = 0.0

T_sim = 15.0

def model(t, y):
    CA, CB, CC, CD = y
    if t <= 5.0:
        T = 400.0
    else:
        T = 400.0 - (400.0 - 350.0) * (t - 5.0) / (T_sim - 5.0)
    r1 = 1.5e8 * np.exp(-10000.0 / T) * CA * CB
    r2 = 2.0e9 * np.exp(-12000.0 / T) * CC * CD
    r_net = r1 - r2
    dCAdt = -r_net
    dCBdt = -r_net
    dCCdt = r_net
    dCDdt = r_net
    return [dCAdt, dCBdt, dCCdt, dCDdt]

sol = solve_ivp(
    fun=model,
    t_span=(0.0, T_sim),
    y0=[CA0, CB0, CC0, CD0],
    max_step=0.01,
    dense_output=True
    )

t_min = np.linspace(0.0, T_sim, 300)
CA = sol.sol(t_min)[0]
CB = sol.sol(t_min)[1]
CC = sol.sol(t_min)[2]
CD = sol.sol(t_min)[3]

T = np.where(t_min <= 5.0, 400.0, 400.0 - (400.0 - 350.0) * (t_min - 5.0) / (T_sim - 5.0))
XA = (CA0 - CA) / CA0

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 8))

ax1.plot(t_min, T, label="T (K)", color='red')
ax1.set_xlabel("Tempo (min)")
ax1.set_ylabel("Temperatura (K)")
ax1.legend()
ax1.grid(True)

ax2.plot(t_min, CA, label="CA", color='blue')
ax2.plot(t_min, CB, label="CB", color='green')
ax2.plot(t_min, CC, label="CC", color='orange')
ax2.plot(t_min, CD, label="CD", color='purple')
ax2.set_xlabel("Tempo (min)")
ax2.set_ylabel("Concentração (mol/L)")
ax2.legend()
ax2.grid(True)

ax3.plot(t_min, XA, label="XA", color='black')
ax3.set_xlabel("Tempo (min)")
ax3.set_ylabel("Conversão de A")
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.savefig("figuras/questao3_reator.png", dpi=300)

with open("figuras/questao3_reator.dat", "w", encoding="utf-8") as f:
    f.write("tempo_min T_K CA CB CC CD XA\n")
    for t, temp, ca, cb, cc, cd, xa in zip(t_min, T, CA, CB, CC, CD, XA):
        f.write(f"{t:.6f} {temp:.6f} {ca:.6f} {cb:.6f} {cc:.6f} {cd:.6f} {xa:.6f}\n")

print("T final:", T[-1])
print("CA final:", CA[-1])
print("CB final:", CB[-1])
print("CC final:", CC[-1])
print("CD final:", CD[-1])
print("XA final:", XA[-1])
