import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# Parâmetros do processo
Ca_in = 0.05  # mol/L
Cb_in = 0.0   # mol/L
Cc_in = 0.0   # mol/L
Cd_in = 0.0   # mol/L 

Qin = 250.0   # L/min
Qout = 250.0  # L/min (assumindo estado estacionário)
V = 1000.0    # L
k1 = 0.5      # min⁻¹
k2 = 0.2      # min⁻¹
k3 = 0.2      # min⁻¹·L·mol⁻¹

# Tempo de residência
tau = V / Qin  # min

def steady_state_equations(concentrations):
    """
    Sistema de equações algébricas para o estado estacionário
    do reator de Van der Vusse
    """
    Ca, Cb, Cc, Cd = concentrations
    
    # Em estado estacionário, dC/dt = 0
    # Equações algébricas resultantes:
    
    eq1 = Qin*Ca_in - Qout*Ca - k1*Ca*V - 2*k3*Ca**2*V
    eq2 = Qin*Cb_in - Qout*Cb + k1*Ca*V - k2*Cb*V
    eq3 = Qin*Cc_in - Qout*Cc + k2*Cb*V
    eq4 = Qin*Cd_in - Qout*Cd + k3*Ca**2*V
    
    return [eq1, eq2, eq3, eq4]

# Estimativas iniciais (valores próximos do esperado)
initial_guess = [0.02, 0.015, 0.01, 0.005]  # Ca, Cb, Cc, Cd

# Resolver o sistema de equações não-lineares
solution = fsolve(steady_state_equations, initial_guess)
Ca_ss, Cb_ss, Cc_ss, Cd_ss = solution

# Verificar se a solução está correta
residuals = steady_state_equations(solution)

# Conversão de A no estado estacionário
XA_ss = (Ca_in - Ca_ss) / Ca_in

print("RESULTADOS DO ESTADO ESTACIONÁRIO")
print("="*50)
print(f"Tempo de residência: {tau:.2f} min")
print()
print("Concentrações em estado estacionário:")
print(f"CA = {Ca_ss:.6f} mol/L")
print(f"CB = {Cb_ss:.6f} mol/L")
print(f"CC = {Cc_ss:.6f} mol/L") 
print(f"CD = {Cd_ss:.6f} mol/L")
print()
print(f"Conversão de A: {XA_ss:.4f} ({XA_ss*100:.2f}%)")
print()
print("Verificação da solução (resíduos das equações):")
print(f"Resíduo eq1: {residuals[0]:.2e}")
print(f"Resíduo eq2: {residuals[1]:.2e}")
print(f"Resíduo eq3: {residuals[2]:.2e}")
print(f"Resíduo eq4: {residuals[3]:.2e}")

# Análise de sensibilidade - efeito do tempo de residência
tau_range = np.linspace(0.5, 10.0, 100)
Ca_sens = []
Cb_sens = []
Cc_sens = []
Cd_sens = []
XA_sens = []

for tau_i in tau_range:
    Q_i = V / tau_i
    
    def equations_sens(concentrations):
        Ca, Cb, Cc, Cd = concentrations
        
        eq1 = Q_i*Ca_in - Q_i*Ca - k1*Ca*V - 2*k3*Ca**2*V
        eq2 = Q_i*Cb_in - Q_i*Cb + k1*Ca*V - k2*Cb*V
        eq3 = Q_i*Cc_in - Q_i*Cc + k2*Cb*V
        eq4 = Q_i*Cd_in - Q_i*Cd + k3*Ca**2*V
        
        return [eq1, eq2, eq3, eq4]
    
    try:
        sol_sens = fsolve(equations_sens, initial_guess)
        Ca_sens.append(sol_sens[0])
        Cb_sens.append(sol_sens[1])
        Cc_sens.append(sol_sens[2])
        Cd_sens.append(sol_sens[3])
        XA_sens.append((Ca_in - sol_sens[0]) / Ca_in)
    except:
        Ca_sens.append(np.nan)
        Cb_sens.append(np.nan)
        Cc_sens.append(np.nan)
        Cd_sens.append(np.nan)
        XA_sens.append(np.nan)

# Gráficos
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Concentrações vs tempo de residência
ax1.plot(tau_range, Ca_sens, label="CA", color='blue', linewidth=2)
ax1.plot(tau_range, Cc_sens, label="CC", color='orange', linewidth=2)
ax1.axvline(x=tau, color='red', linestyle='--', alpha=0.7, label=f'τ atual = {tau:.1f} min')
ax1.set_xlabel("Tempo de residência (min)")
ax1.set_ylabel("Concentração CA e CC (mol/L)")
ax1.legend(loc='upper left')
ax1.grid(True)

# Segunda escala para CB e CD
ax1_twin = ax1.twinx()
ax1_twin.plot(tau_range, Cb_sens, label="CB", color='green', linewidth=2)
ax1_twin.plot(tau_range, Cd_sens, label="CD", color='purple', linewidth=2)
ax1_twin.set_ylabel("Concentração CB e CD (mol/L)")
ax1_twin.legend(loc='upper right')

ax1.set_title("Concentrações em Estado Estacionário vs Tempo de Residência")

# Conversão vs tempo de residência
ax2.plot(tau_range, XA_sens, color='black', linewidth=2)
ax2.axvline(x=tau, color='red', linestyle='--', alpha=0.7, label=f'τ atual = {tau:.1f} min')
ax2.axhline(y=XA_ss, color='red', linestyle=':', alpha=0.7, label=f'XA atual = {XA_ss:.3f}')
ax2.set_xlabel("Tempo de residência (min)")
ax2.set_ylabel("Conversão de A")
ax2.legend()
ax2.grid(True)
ax2.set_title("Conversão de A vs Tempo de Residência")

plt.tight_layout()
plt.savefig("van_der_vusse_estado_estacionario.png", dpi=300)
plt.show()

# Gráfico destacando CB
fig_cb, ax_cb = plt.subplots(1, 1, figsize=(8, 6))
ax_cb.plot(tau_range, Cb_sens, color='green', linewidth=3, label='CB em estado estacionário')
ax_cb.axvline(x=tau, color='red', linestyle='--', alpha=0.7, linewidth=2, label=f'τ atual = {tau:.1f} min')
ax_cb.axhline(y=Cb_ss, color='red', linestyle=':', alpha=0.7, linewidth=2, label=f'CB atual = {Cb_ss:.4f} mol/L')
ax_cb.scatter([tau], [Cb_ss], color='red', s=100, zorder=5)
ax_cb.set_xlabel("Tempo de residência (min)")
ax_cb.set_ylabel("Concentração CB (mol/L)")
ax_cb.set_title("Concentração de B em Estado Estacionário")
ax_cb.legend()
ax_cb.grid(True)
plt.savefig("cb_estado_estacionario.png", dpi=300)
plt.show()

# Salvar dados
with open("estado_estacionario.dat", "w", encoding="utf-8") as f:
    f.write("# Resultados do Estado Estacionário - Reator Van der Vusse\n")
    f.write("# Parâmetros:\n")
    f.write(f"# Ca_in = {Ca_in} mol/L\n")
    f.write(f"# Q = {Qin} L/min\n")
    f.write(f"# V = {V} L\n")
    f.write(f"# tau = {tau} min\n")
    f.write(f"# k1 = {k1} min⁻¹\n")
    f.write(f"# k2 = {k2} min⁻¹\n")
    f.write(f"# k3 = {k3} min⁻¹·L·mol⁻¹\n")
    f.write("#\n")
    f.write("# Concentrações em estado estacionário:\n")
    f.write(f"CA_ss = {Ca_ss:.6f} # mol/L\n")
    f.write(f"CB_ss = {Cb_ss:.6f} # mol/L\n")
    f.write(f"CC_ss = {Cc_ss:.6f} # mol/L\n")
    f.write(f"CD_ss = {Cd_ss:.6f} # mol/L\n")
    f.write(f"XA_ss = {XA_ss:.6f} # conversão\n")
    f.write("#\n")
    f.write("# Análise de sensibilidade:\n")
    f.write("tau_min CA CB CC CD XA\n")
    for tau_i, ca, cb, cc, cd, xa in zip(tau_range, Ca_sens, Cb_sens, Cc_sens, Cd_sens, XA_sens):
        if not np.isnan(cb):
            f.write(f"{tau_i:.4f} {ca:.6f} {cb:.6f} {cc:.6f} {cd:.6f} {xa:.6f}\n")