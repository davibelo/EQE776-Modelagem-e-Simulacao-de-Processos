import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parâmetros do processo
Ca_in = 0.05  # mol/L
Qin = 250.0   # L/min
V = 1000.0    # L
k1 = 0.5      # min⁻¹
k2 = 0.2      # min⁻¹
k3 = 0.2      # min⁻¹·L·mol⁻¹

# Condições iniciais
Ca0 = 0.05    # mol/L
Cb0 = 0.0     # mol/L
Cc0 = 0.0     # mol/L
Cd0 = 0.0     # mol/L

# Tempo de residência
tau = V / Qin  # min

# Tempo de simulação
T_sim = 20.0  # min

def model(t, y):
    Ca, Cb, Cc, Cd = y
    
    # Taxas de reação
    r1 = k1 * Ca
    r2 = k2 * Cb
    r3 = k3 * Ca**2
    
    # Equações diferenciais do reator CSTR
    dCadt = (Qin/V) * (Ca_in - Ca) - r1 - 2*r3
    dCbdt = (Qin/V) * (0 - Cb) + r1 - r2
    dCcdt = (Qin/V) * (0 - Cc) + r2
    dCddt = (Qin/V) * (0 - Cd) + r3
    
    return [dCadt, dCbdt, dCcdt, dCddt]

# Solução do sistema de EDOs
sol = solve_ivp(
    fun=model,
    t_span=(0.0, T_sim),
    y0=[Ca0, Cb0, Cc0, Cd0],
    max_step=0.01,
    dense_output=True
)

# Vetores de tempo e concentrações
t_min = np.linspace(0.0, T_sim, 300)
Ca = sol.sol(t_min)[0]
Cb = sol.sol(t_min)[1]
Cc = sol.sol(t_min)[2]
Cd = sol.sol(t_min)[3]

# Conversão de A
XA = (Ca0 - Ca) / Ca0

# Gráficos
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

# Concentrações vs tempo
ax1.plot(t_min, Ca, label="CA", color='blue')
ax1.plot(t_min, Cb, label="CB", color='green')
ax1.plot(t_min, Cc, label="CC", color='orange')
ax1.plot(t_min, Cd, label="CD", color='purple')
ax1.set_xlabel("Tempo (min)")
ax1.set_ylabel("Concentração (mol/L)")
ax1.legend()
ax1.grid(True)
ax1.set_title("Concentrações no Reator de Van der Vusse")

# Conversão vs tempo
ax2.plot(t_min, XA, label="XA", color='black')
ax2.set_xlabel("Tempo (min)")
ax2.set_ylabel("Conversão de A")
ax2.legend()
ax2.grid(True)
ax2.set_title("Conversão de A")

plt.tight_layout()
plt.savefig("exercicios sala de aula/aula 08/van_der_vusse_reator.png", dpi=300)
plt.show()

# Salvar dados
with open("exercicios sala de aula/aula 08/van_der_vusse_reator.dat", "w", encoding="utf-8") as f:
    f.write("tempo_min CA CB CC CD XA\n")
    for t, ca, cb, cc, cd, xa in zip(t_min, Ca, Cb, Cc, Cd, XA):
        f.write(f"{t:.6f} {ca:.6f} {cb:.6f} {cc:.6f} {cd:.6f} {xa:.6f}\n")

# Valores finais
print("Tempo de residência:", tau, "min")
print("Concentrações finais:")
print("CA final:", Ca[-1])
print("CB final:", Cb[-1])
print("CC final:", Cc[-1])
print("CD final:", Cd[-1])
print("XA final:", XA[-1])