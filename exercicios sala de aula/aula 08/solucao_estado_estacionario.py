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

    r1 = k1 * Ca
    r2 = k2 * Cb
    r3 = k3 * Ca**2
    
    eq1 = Qin*Ca_in - Qout*Ca - r1*V - 2*r3*V
    eq2 = Qin*Cb_in - Qout*Cb + r1*V - r2*V
    eq3 = Qin*Cc_in - Qout*Cc + r2*V
    eq4 = Qin*Cd_in - Qout*Cd + r3*V
    
    return [eq1, eq2, eq3, eq4]

# Estimativas iniciais (valores próximos do esperado)
initial_guess = [Ca_in, Cb_in, Cc_in, Cd_in]  # Ca, Cb, Cc, Cd

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

