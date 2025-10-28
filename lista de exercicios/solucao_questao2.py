import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

F1 = 10.0
xA1 = 0.6
xB1 = 0.0
xC1 = 0.4
F2 = 8.0
xA2 = 0.0
xB2 = 0.7
xC2 = 0.3
rhoA = 1200.0
rhoB = 1400.0
rhoC = 1000.0
A = 0.2
k = 0.02
mA0 = 20.0
mB0 = 20.0
mC0 = 40.0

T_sim = 60.0

def model(t, y):
    mA, mB, mC = y
    m_total = mA + mB + mC
    xA = mA / m_total
    xB = mB / m_total
    xC = mC / m_total
    rho3 = 1.0 / (xA / rhoA + xB / rhoB + xC / rhoC)
    V = mA / rhoA + mB / rhoB + mC / rhoC
    h = V / A
    F3 = rho3 * k * np.sqrt(max(h, 0.0))
    dmAdt = F1 * xA1 + F2 * xA2 - F3 * xA
    dmBdt = F1 * xB1 + F2 * xB2 - F3 * xB
    dmCdt = F1 * xC1 + F2 * xC2 - F3 * xC
    return [dmAdt, dmBdt, dmCdt]

sol = solve_ivp(
    fun=model,
    t_span=(0.0, T_sim),
    y0=[mA0, mB0, mC0],
    max_step=0.01,
    dense_output=True
    )

t_min = np.linspace(0.0, T_sim, 200)
mA = sol.sol(t_min)[0]
mB = sol.sol(t_min)[1]
mC = sol.sol(t_min)[2]

m_total = mA + mB + mC
xA = mA / m_total
xB = mB / m_total
xC = mC / m_total
V = mA / rhoA + mB / rhoB + mC / rhoC
h = V / A

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))

ax1.plot(t_min, h, label="h (m)", color='blue')
ax1.set_xlabel("Tempo (min)")
ax1.set_ylabel("Nível (m)")
ax1.legend()
ax1.grid(True)

ax2.plot(t_min, xA, label="xA", color='red')
ax2.plot(t_min, xB, label="xB", color='green')
ax2.plot(t_min, xC, label="xC", color='purple')
ax2.set_xlabel("Tempo (min)")
ax2.set_ylabel("Fração mássica")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("figuras/questao2_tanque.png", dpi=300)

with open("figuras/questao2_tanque.dat", "w", encoding="utf-8") as f:
    f.write("tempo_min h_m xA xB xC\n")
    for t, hval, xa, xb, xc in zip(t_min, h, xA, xB, xC):
        f.write(f"{t:.6f} {hval:.6f} {xa:.6f} {xb:.6f} {xc:.6f}\n")

print("h final:", h[-1])
print("xA final:", xA[-1])
print("xB final:", xB[-1])
print("xC final:", xC[-1])
