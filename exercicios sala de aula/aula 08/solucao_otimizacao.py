import numpy as np
from scipy.optimize import fsolve, minimize_scalar

# ==============================================================================
# 1. PARÂMETROS DO PROCESSO
# ==============================================================================
Ca_in = 0.05  # mol/L
Cb_in = 0.0   # mol/L
Cc_in = 0.0   # mol/L
Cd_in = 0.0   # mol/L

V = 1000.0           # L
k1 = 0.5             # min⁻¹
k2 = 0.2             # min⁻¹
k3 = 0.2             # min⁻¹·L·mol⁻¹

# Limites para a otimização de Qin
Qin_min = 150.0      # L/min
Qin_max = 500.0      # L/min

# ==============================================================================
# 2. MODELO DO PROCESSO
# ==============================================================================
def steady_state_equations(concentrations, Qin):
    """
    Sistema de equações algébricas para o estado estacionário
    do reator de Van der Vusse
    """
    Qout = Qin  # condição de estado estacionário (volume constante)
    Ca, Cb, Cc, Cd = concentrations

    # Taxas de reação
    r1 = k1 * Ca
    r2 = k2 * Cb
    r3 = k3 * Ca**2

    # Balanços molares (Entrada - Saída + Geração = 0)
    eq1 = Qin*Ca_in - Qout*Ca - r1*V - 2*r3*V
    eq2 = Qin*Cb_in - Qout*Cb + r1*V - r2*V
    eq3 = Qin*Cc_in - Qout*Cc + r2*V
    eq4 = Qin*Cd_in - Qout*Cd + r3*V

    return [eq1, eq2, eq3, eq4]

def solve_steady_state(Qin):
    """
    Resolve o sistema de equações para um dado Qin e retorna os resultados.
    """
    # Chute inicial fixo (Feed) para garantir consistência durante a otimização
    initial_guess = [Ca_in, Cb_in, Cc_in, Cd_in]

    solution = fsolve(steady_state_equations, initial_guess, args=(Qin,))
    Ca, Cb, Cc, Cd = solution
    
    residuals = steady_state_equations(solution, Qin)
    tau = V / Qin if Qin != 0 else np.inf
    XA = (Ca_in - Ca) / Ca_in if Ca_in != 0 else 0

    return {
        "Qin": Qin,
        "Ca": Ca,
        "Cb": Cb,
        "Cc": Cc,
        "Cd": Cd,
        "XA": XA,
        "tau": tau
    }

# ==============================================================================
# 3. CONFIGURAÇÃO DA OTIMIZAÇÃO
# ==============================================================================
def objective_function(Qin):
    """
    Função objetivo a ser MINIMIZADA.
    Como queremos MAXIMIZAR Cb, retornamos o negativo de Cb (-Cb).
    """
    # Otimizadores às vezes passam arrays de tamanho 1, garantimos float
    Qin_val = float(Qin)
    
    # Resolve o estado estacionário para o Qin atual
    results = solve_steady_state(Qin_val)
    
    # Retornamos -Cb para que o 'minimize' encontre o máximo de Cb
    return -results['Cb']

def print_result(title, result):
    """Auxiliar para imprimir os resultados"""
    print(title)
    print("=" * len(title))
    print(f"Vazão de Alimentação (Qin): {result['Qin']:.4f} L/min")
    print(f"Tempo de residência (tau):  {result['tau']:.4f} min")
    print("-" * 30)
    print("Concentrações Ótimas:")
    print(f"  CA = {result['Ca']:.6f} mol/L")
    print(f"  CB = {result['Cb']:.6f} mol/L  <-- OTIMIZADO")
    print(f"  CC = {result['Cc']:.6f} mol/L")
    print(f"  CD = {result['Cd']:.6f} mol/L")
    print("-" * 30)
    print(f"Conversão de A (XA):        {result['XA']:.4f} ({result['XA']*100:.2f}%)")
    print()

# ==============================================================================
# 4. EXECUÇÃO
# ==============================================================================
if __name__ == "__main__":
    print(f"Iniciando otimização de Cb variando Qin entre {Qin_min} e {Qin_max} L/min...")
    
    # Executa a otimização escalar
    # method='bounded' é ideal para intervalos finitos
    opt_result = minimize_scalar(
        objective_function, 
        bounds=(Qin_min, Qin_max), 
        method='bounded'
    )

    if opt_result.success:
        # Recupera o Qin ótimo encontrado
        best_Qin = opt_result.x
        
        # Recalcula o estado completo para esse ponto ótimo
        final_state = solve_steady_state(best_Qin)
        
        print("\nOTIMIZAÇÃO CONCLUÍDA COM SUCESSO!")
        print_result("RESULTADOS DO PONTO ÓTIMO", final_state)
    else:
        print("\nA otimização falhou.")
        print(opt_result)
