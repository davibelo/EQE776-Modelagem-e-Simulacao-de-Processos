# Bibliotecas para cálculo numérico, visualização e integração de EDOs
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
import os

# Detecção do diretório base para salvar resultados
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

# Caminhos para salvar figuras e resultados
FIGURE_PATHS = {
    "evolucao_temporal": os.path.join(BASE_DIR, "evolucao_temporal.png"),
    "perfil_espacial": os.path.join(BASE_DIR, "perfil_espacial.png"),
    "grafico_3d": os.path.join(BASE_DIR, "grafico_3d.png"),
}
RESULTS_PATH = os.path.join(BASE_DIR, "resultados_pfr.txt")

# Parâmetros do reator PFR
L = 1.0          # Comprimento adimensional
Calim = 1.0      # Concentração de alimentação
Pe = 15.0        # Número de Péclet (convecção/dispersão)
Da = 1.0         # Número de Damköhler (reação/convecção)
Np = 50          # Número de pontos internos

DL = L / Np      # Espaçamento entre pontos

# Sistema de EDOs para PFR com dispersão axial
def sistema_odes(t, C_interior):
    # Condição de contorno de Danköhler na entrada
    C0 = (Calim + C_interior[0]/(Pe*2*DL)) / (1 + 1/(Pe*2*DL))

    # Array completo com pontos virtuais
    C = np.zeros(Np + 2)
    C[0] = C0
    C[1:Np+1] = C_interior
    C[Np+1] = C_interior[-1]  # Condição de Neumann na saída

    dCdt = np.zeros(Np)

    # Balanço em cada ponto: dispersão - convecção - reação
    for i in range(1, Np + 1):
        dC_dz = (C[i+1] - C[i-1]) / (2 * DL)
        d2C_dz2 = (C[i+1] - 2*C[i] + C[i-1]) / (DL**2)
        dCdt[i-1] = (1/Pe) * d2C_dz2 - dC_dz - Da * C[i]

    return dCdt

# Configuração da simulação temporal
t_span = (0, 2)
t_eval = np.linspace(0, 2, 2000)

print(f"Simulando PFR com dispersão axial...")
print(f"  Np = {Np}")
print(f"  L = {L}")
print(f"  Pe = {Pe}")
print(f"  Da = {Da}")
print(f"  Calim = {Calim}")
print(f"  DL = {DL:.6f}")

# Condição inicial: reator vazio
C0_interior = np.zeros(Np)

# Resolução do sistema de EDOs
sol = solve_ivp(
    sistema_odes,
    t_span,
    C0_interior,
    t_eval=t_eval,
    method='BDF',
    rtol=1e-6,
    atol=1e-8
)

if not sol.success:
    print(f"Aviso: {sol.message}")
else:
    print("Simulação concluída com sucesso!")

# Reconstrução do perfil completo incluindo pontos virtuais
C_completo = np.zeros((Np + 2, len(sol.t)))
for j in range(len(sol.t)):
    C0 = (Calim + sol.y[0, j]/(Pe*2*DL)) / (1 + 1/(Pe*2*DL))
    C_completo[0, j] = C0
    C_completo[1:Np+1, j] = sol.y[:, j]
    C_completo[Np+1, j] = sol.y[-1, j]

z = np.linspace(0, L, Np + 2)

# Índices para posições de interesse
idx_in = 1
idx_um_quarto = int(1 + Np * 0.25)
idx_meio = int(1 + Np * 0.5)
idx_tres_quartos = int(1 + Np * 0.75)
idx_out = Np

print("\nGerando gráficos...")

# Gráfico 1: Evolução temporal em diferentes posições
fig1, ax1 = plt.subplots(figsize=(12, 7))

posicoes = [
    (idx_in, z[idx_in], 'black'),
    (idx_um_quarto, z[idx_um_quarto], 'red'),
    (idx_meio, z[idx_meio], 'blue'),
    (idx_tres_quartos, z[idx_tres_quartos], 'green'),
    (idx_out, z[idx_out], 'cyan'),
]

for idx, pos, color in posicoes:
    ax1.plot(sol.t, C_completo[idx, :], linewidth=2.5, color=color,
             label=f'C para L={pos:.2f}')

ax1.set_xlabel('t\' (tempo adimensional)', fontsize=13)
ax1.set_ylabel('C (concentração adimensional)', fontsize=13)
ax1.set_title(f'Evolução Temporal da Concentração em Diferentes Posições\nPe = {Pe}, Da = {Da}, Np = {Np}',
              fontsize=14, fontweight='bold')
ax1.legend(fontsize=11, loc='right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0, 2])
ax1.set_ylim([0, 1.0])
fig1.tight_layout()
fig1.savefig(FIGURE_PATHS["evolucao_temporal"], dpi=300, bbox_inches='tight')

# Gráfico 2: Perfil espacial em diferentes tempos
fig2, ax2 = plt.subplots(figsize=(12, 7))

tempos = [0, 0.25, 0.5, 1.0, 2.0]
colors = ['black', 'red', 'blue', 'green', 'cyan']

for tempo, color in zip(tempos, colors):
    idx_time = np.argmin(np.abs(sol.t - tempo))
    ax2.plot(z, C_completo[:, idx_time], linewidth=2.5, color=color,
             label=f't\' = {tempo:.2f}')

ax2.set_xlabel('L (posição)', fontsize=13)
ax2.set_ylabel('C (concentração adimensional)', fontsize=13)
ax2.set_title(f'Perfil Espacial de Concentração em Diferentes Tempos\nPe = {Pe}, Da = {Da}, Np = {Np}',
              fontsize=14, fontweight='bold')
ax2.legend(fontsize=11, loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim([0, L])
ax2.set_ylim([0, 1.0])
fig2.tight_layout()
fig2.savefig(FIGURE_PATHS["perfil_espacial"], dpi=300, bbox_inches='tight')

# Gráfico 3: Superfície 3D (posição × tempo × concentração)
fig3 = plt.figure(figsize=(14, 10))
ax3 = fig3.add_subplot(111, projection='3d')

T, Z = np.meshgrid(sol.t, z)
C_plot = C_completo

surf = ax3.plot_surface(Z, T, C_plot, cmap='viridis', edgecolor='none', alpha=0.9)

ax3.set_xlabel('L (posição)', fontsize=12, labelpad=10)
ax3.set_ylabel("t' (tempo adimensional)", fontsize=12, labelpad=10)
ax3.set_zlabel('C (concentração adimensional)', fontsize=12, labelpad=10)
ax3.set_title(f'Superfície 3D: Concentração em Função do Tempo e Posição\nPe = {Pe}, Da = {Da}, Np = {Np}',
              fontsize=14, fontweight='bold', pad=20)

fig3.colorbar(surf, ax=ax3, shrink=0.5, aspect=5, label='Concentração')

ax3.view_init(elev=20, azim=-60)

fig3.tight_layout()
fig3.savefig(FIGURE_PATHS["grafico_3d"], dpi=300, bbox_inches='tight')

plt.show()

# Preparação dos resultados textuais
result_lines = []
result_lines.append("="*70)
result_lines.append("RESULTADOS DA SIMULAÇÃO DO PFR ISOTÉRMICO COM DISPERSÃO AXIAL")
result_lines.append("="*70)
result_lines.append(f"Parâmetros:")
result_lines.append(f"  L (comprimento) = {L}")
result_lines.append(f"  Calim = {Calim}")
result_lines.append(f"  Pe (Péclet) = {Pe}")
result_lines.append(f"  Da (Damköhler) = {Da}")
result_lines.append(f"  Np (número de pontos internos) = {Np}")
result_lines.append(f"  DL = {DL:.6f}")
result_lines.append("")
result_lines.append("Discretização:")
result_lines.append(f"  i=0 (ponto virtual): condição Danköhler")
result_lines.append(f"  i=1 a i={Np}: pontos internos com EDOs")
result_lines.append(f"  i={Np+1} (ponto virtual): condição Neumann")
result_lines.append("")
result_lines.append("Equações:")
result_lines.append(f"  dC_i/dt = (1/Pe)*(C_{{i+1}}-2C_i+C_{{i-1}})/DL² - (C_{{i+1}}-C_{{i-1}})/(2*DL) - Da*C_i")
result_lines.append(f"  C_0 = Calim + (1/Pe)*(C_1-C_0)/(2*DL)")
result_lines.append(f"  C_{Np+1} = C_{Np}")
result_lines.append("="*70)
result_lines.append("")

result_lines.append("CONCENTRAÇÕES EM DIFERENTES POSIÇÕES (t' = 2.0):")
for idx, pos, _ in posicoes:
    C_final = C_completo[idx, -1]
    result_lines.append(f"  L = {pos:.4f}: C = {C_final:.6f}")

result_lines.append("")
conversao = (Calim - C_completo[idx_out, -1]) / Calim * 100
result_lines.append(f"CONVERSÃO NA SAÍDA: {conversao:.2f}%")
result_lines.append("")

result_lines.append("EVOLUÇÃO TEMPORAL NA SAÍDA (L ≈ 1.0):")
for tempo in [0, 0.25, 0.5, 1.0, 1.5, 2.0]:
    idx_time = np.argmin(np.abs(sol.t - tempo))
    C_out_t = C_completo[idx_out, idx_time]
    result_lines.append(f"  t' = {tempo:4.2f}: C = {C_out_t:.6f}")

print('\n'.join(result_lines))

# Salvar resultados em arquivo
with open(RESULTS_PATH, 'w', encoding='utf-8') as results_file:
    results_file.write('\n'.join(result_lines))

print(f"\nResultados salvos em: {RESULTS_PATH}")
print("Gráficos salvos:")
for key, path in FIGURE_PATHS.items():
    print(f"  - {key}: {path}")
