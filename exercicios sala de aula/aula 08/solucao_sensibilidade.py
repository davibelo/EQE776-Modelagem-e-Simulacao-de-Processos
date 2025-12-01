import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# Parâmetros do processo
Ca_in = 0.05  # mol/L
Cb_in = 0.0   # mol/L
Cc_in = 0.0   # mol/L
Cd_in = 0.0   # mol/L

Qin_default = 250.0  # L/min
V = 1000.0           # L
k1 = 0.5             # min⁻¹
k2 = 0.2             # min⁻¹
k3 = 0.2             # min⁻¹·L·mol⁻¹

# Configuração da análise de sensibilidade
aq_min = 150.0       # L/min
aq_max = 500.0       # L/min
num_casos = 20       # quantidade de cenários

def steady_state_equations(concentrations, Qin):
    """
    Sistema de equações algébricas para o estado estacionário
    do reator de Van der Vusse
    """
    Qout = Qin  # condição de estado estacionário
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


def solve_steady_state(Qin, initial_guess=None):
    """
    Resolve o sistema de equações para um dado Qin
    """
    if initial_guess is None:
        initial_guess = [Ca_in, Cb_in, Cc_in, Cd_in]

    solution = fsolve(steady_state_equations, initial_guess, args=(Qin,))
    Ca, Cb, Cc, Cd = solution

    residuals = steady_state_equations(solution, Qin)
    tau = V / Qin
    XA = (Ca_in - Ca) / Ca_in

    return {
        "Qin": Qin,
        "Ca": Ca,
        "Cb": Cb,
        "Cc": Cc,
        "Cd": Cd,
        "XA": XA,
        "tau": tau,
        "residuals": residuals,
        "solution_vector": solution,
    }


def print_result(title, result):
    """
    Imprime os resultados do estado estacionário de forma formatada
    """
    print(title)
    print("=" * len(title))
    print(f"Qin: {result['Qin']:.2f} L/min")
    print(f"Tempo de residência: {result['tau']:.2f} min")
    print()
    print("Concentrações em estado estacionário:")
    print(f"CA = {result['Ca']:.6f} mol/L")
    print(f"CB = {result['Cb']:.6f} mol/L")
    print(f"CC = {result['Cc']:.6f} mol/L")
    print(f"CD = {result['Cd']:.6f} mol/L")
    print()
    print(f"Conversão de A: {result['XA']:.4f} ({result['XA'] * 100:.2f}%)")
    print()
    print("Verificação da solução (resíduos das equações):")
    print(f"Resíduo eq1: {result['residuals'][0]:.2e}")
    print(f"Resíduo eq2: {result['residuals'][1]:.2e}")
    print(f"Resíduo eq3: {result['residuals'][2]:.2e}")
    print(f"Resíduo eq4: {result['residuals'][3]:.2e}")
    print()


# Resolver para Qin padrão
base_result = solve_steady_state(Qin_default)
print_result("RESULTADOS DO ESTADO ESTACIONÁRIO (Qin padrão)", base_result)

# Análise de sensibilidade com variação de Qin
qin_values = np.linspace(aq_min, aq_max, num_casos)
sensitivity_results = []
current_guess = base_result["solution_vector"]

for Qin in qin_values:
    result = solve_steady_state(Qin, current_guess)
    sensitivity_results.append(result)
    current_guess = result["solution_vector"]

# Exibir tabela de sensibilidade
print("ANÁLISE DE SENSIBILIDADE (variação de Qin)")
print("=" * 60)
print(f"{'Qin (L/min)':>12} {'Ca (mol/L)':>12} {'Cb (mol/L)':>12} {'Cc (mol/L)':>12} {'Cd (mol/L)':>12} {'XA':>8} {'tau (min)':>12}")
for res in sensitivity_results:
    print(
        f"{res['Qin']:12.2f} {res['Ca']:12.6f} {res['Cb']:12.6f} {res['Cc']:12.6f} "
        f"{res['Cd']:12.6f} {res['XA']:8.4f} {res['tau']:12.4f}"
    )

# Gráfico de sensibilidade: Cb vs Qin
plt.figure(figsize=(8, 5))
plt.plot([res["Qin"] for res in sensitivity_results], [res["Cb"] for res in sensitivity_results], marker="o")
plt.xlabel("Qin (L/min)")
plt.ylabel("Cb (mol/L)")
plt.title("Sensibilidade de Cb em função de Qin")
plt.grid(True)
plt.tight_layout()
plt.show()