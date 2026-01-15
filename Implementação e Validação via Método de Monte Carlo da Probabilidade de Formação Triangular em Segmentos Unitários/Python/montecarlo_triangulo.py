import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon

# Configuração para melhor visualização
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def simulacao_monte_carlo(n_simulacoes=200000, seed=42):
    """
    Realiza simulações de Monte Carlo para estimar a probabilidade
    de que três segmentos formados por dois cortes aleatórios
    em um segmento unitário satisfaçam a desigualdade triangular.
    """
    np.random.seed(seed)
    
    # Gerar dois pontos de corte aleatórios
    cortes = np.random.random((n_simulacoes, 2))
    
    # Ordenar os cortes para obter três segmentos
    cortes_ordenados = np.sort(cortes, axis=1)
    
    # Calcular comprimentos dos três segmentos: a, b, c
    a = cortes_ordenados[:, 0]  # Primeiro segmento (0 até primeiro corte)
    b = cortes_ordenados[:, 1] - cortes_ordenados[:, 0]  # Segmento médio
    c = 1 - cortes_ordenados[:, 1]  # Último segmento (segundo corte até 1)
    
    # Verificar desigualdade triangular para cada triângulo possível
    # Para formar triângulo, cada segmento deve ser menor que a soma dos outros dois
    forma_triangulo = (a < b + c) & (b < a + c) & (c < a + b)
    
    # Probabilidade experimental
    p_estimada = np.mean(forma_triangulo)
    erro_padrao = np.sqrt(p_estimada * (1 - p_estimada) / n_simulacoes)
    
    # Coletar dados dos triângulos formados
    triangulos = np.column_stack((a[forma_triangulo], b[forma_triangulo], c[forma_triangulo]))
    
    return {
        'p_estimada': p_estimada,
        'erro_padrao': erro_padrao,
        'forma_triangulo': forma_triangulo,
        'triangulos': triangulos,
        'comprimentos': (a, b, c),
        'cortes': cortes
    }

def plot_histogramas_comprimentos(resultados):
    """Histogramas dos comprimentos dos segmentos."""
    a, b, c = resultados['comprimentos']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for ax, dados, label in zip(axes, [a, b, c], ['a', 'b', 'c']):
        ax.hist(dados, bins=50, density=True, alpha=0.7, edgecolor='black')
        ax.set_xlabel(f'Comprimento {label}')
        ax.set_ylabel('Densidade')
        ax.set_title(f'Distribuição do comprimento {label}')
        
        # Adicionar linha teórica (distribuição uniforme em [0, 1] com média 1/3)
        x = np.linspace(0, 1, 100)
        ax.axvline(1/3, color='red', linestyle='--', label=f'Média teórica: 1/3')
        ax.legend()
    
    plt.suptitle('Distribuição dos Comprimentos dos Segmentos', fontsize=14, y=1.05)
    plt.tight_layout()
    return fig

def plot_diagrama_dispersao_cortes(resultados):
    """Diagrama de dispersão dos pontos de corte."""
    cortes = resultados['cortes']
    forma_triangulo = resultados['forma_triangulo']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plotar pontos que formam triângulo
    scatter1 = ax.scatter(cortes[forma_triangulo, 0], cortes[forma_triangulo, 1], 
                         alpha=0.5, s=10, label='Forma triângulo', color='green')
    
    # Plotar pontos que não formam triângulo
    scatter2 = ax.scatter(cortes[~forma_triangulo, 0], cortes[~forma_triangulo, 1], 
                         alpha=0.1, s=10, label='Não forma triângulo', color='red')
    
    # Adicionar linhas de referência
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='x = y')
    ax.plot([0, 0.5], [0.5, 1], 'b--', alpha=0.3, label='Região teórica de formação')
    
    ax.set_xlabel('Primeiro corte (x)')
    ax.set_ylabel('Segundo corte (y)')
    ax.set_title('Diagrama de Dispersão dos Pontos de Corte')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal', 'box')
    ax.legend(loc='upper left')
    
    # Adicionar texto com a probabilidade
    ax.text(0.02, 0.98, f'P(formar triângulo) = {resultados["p_estimada"]:.4f}',
            transform=ax.transAxes, fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    return fig

def plot_espaco_abc(resultados, n_amostras=2000):
    """Representação 3D da região viável no espaço (a, b, c)."""
    triangulos = resultados['triangulos']
    
    if len(triangulos) == 0:
        return None
    
    # Amostrar para visualização
    if len(triangulos) > n_amostras:
        idx = np.random.choice(len(triangulos), n_amostras, replace=False)
        triangulos_amostra = triangulos[idx]
    else:
        triangulos_amostra = triangulos
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plotar pontos no espaço (a, b, c)
    scatter = ax.scatter(triangulos_amostra[:, 0], 
                        triangulos_amostra[:, 1], 
                        triangulos_amostra[:, 2],
                        c=np.sqrt(triangulos_amostra[:, 0]**2 + 
                                 triangulos_amostra[:, 1]**2 + 
                                 triangulos_amostra[:, 2]**2),
                        cmap='viridis', alpha=0.6, s=20)
    
    # Adicionar ponto do triângulo equilátero (ideal)
    ax.scatter([1/3], [1/3], [1/3], color='red', s=200, marker='*', label='Triângulo equilátero')
    
    # Adicionar plano a + b + c = 1
    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 10)
    X, Y = np.meshgrid(x, y)
    Z = 1 - X - Y
    Z[Z < 0] = np.nan
    
    ax.plot_surface(X, Y, Z, alpha=0.2, color='blue', label='a + b + c = 1')
    
    # Configurações do gráfico
    ax.set_xlabel('Comprimento a')
    ax.set_ylabel('Comprimento b')
    ax.set_zlabel('Comprimento c')
    ax.set_title('Espaço de Comprimentos (a, b, c) que Formam Triângulos')
    ax.legend()
    
    # Adicionar barra de cores
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Norma dos comprimentos')
    
    return fig

def plot_triangulos_quase_equilateros(resultados, n_exemplos=6):
    """Visualização de triângulos formados, mostrando tendência a equiláteros."""
    triangulos = resultados['triangulos']
    
    if len(triangulos) == 0:
        return None
    
    # Selecionar exemplos aleatórios
    if len(triangulos) > n_exemplos:
        idx = np.random.choice(len(triangulos), n_exemplos, replace=False)
        exemplos = triangulos[idx]
    else:
        exemplos = triangulos[:n_exemplos]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (ax, (a, b, c)) in enumerate(zip(axes, exemplos)):
        # Calcular ângulos usando lei dos cossenos
        angulo_A = np.degrees(np.arccos((b**2 + c**2 - a**2) / (2*b*c + 1e-10)))
        angulo_B = np.degrees(np.arccos((a**2 + c**2 - b**2) / (2*a*c + 1e-10)))
        angulo_C = 180 - angulo_A - angulo_B
        
        # Desenhar triângulo
        # Posicionar vértice A na origem, B no eixo x
        A = np.array([0, 0])
        B = np.array([c, 0])
        
        # Calcular posição do vértice C usando lei dos cossenos
        C_x = b * np.cos(np.radians(angulo_A))
        C_y = b * np.sin(np.radians(angulo_A))
        C = np.array([C_x, C_y])
        
        vertices = np.array([A, B, C])
        
        # Criar polígono
        triangulo = Polygon(vertices, closed=True, alpha=0.7, 
                           edgecolor='darkblue', linewidth=2)
        ax.add_patch(triangulo)
        
        # Configurar limites
        ax.set_xlim(-0.1, max(1.1, vertices[:, 0].max() + 0.1))
        ax.set_ylim(-0.1, vertices[:, 1].max() + 0.1)
        ax.set_aspect('equal', 'box')
        
        # Adicionar rótulos
        ax.text(A[0], A[1]-0.05, 'A', fontsize=12, ha='center', va='top')
        ax.text(B[0], B[1]-0.05, 'B', fontsize=12, ha='center', va='top')
        ax.text(C[0], C[1]+0.05, 'C', fontsize=12, ha='center', va='bottom')
        
        # Adicionar comprimentos
        ax.text(np.mean([A[0], B[0]]), np.mean([A[1], B[1]])-0.05, 
                f'{c:.2f}', ha='center', va='top')
        ax.text(np.mean([A[0], C[0]]), np.mean([A[1], C[1]]), 
                f'{b:.2f}', ha='right', va='center')
        ax.text(np.mean([B[0], C[0]]), np.mean([B[1], C[1]]), 
                f'{a:.2f}', ha='left', va='center')
        
        # Calcular desvio do equilátero
        desvio = np.std([a, b, c]) / np.mean([a, b, c])
        
        ax.set_title(f'Exemplo {i+1}\nDesvio relativo: {desvio:.2%}')
    
    # Remover eixos extras se necessário
    for i in range(len(exemplos), len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle('Triângulos Formados - Tendência a Equiláteros', fontsize=16, y=1.02)
    plt.tight_layout()
    return fig

def plot_distribuicao_lados_triangulos(resultados):
    """Distribuição dos lados dos triângulos formados."""
    triangulos = resultados['triangulos']
    
    if len(triangulos) == 0:
        return None
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for i, (ax, lado) in enumerate(zip(axes, ['a', 'b', 'c'])):
        dados = triangulos[:, i]
        ax.hist(dados, bins=50, density=True, alpha=0.7, color=f'C{i}', edgecolor='black')
        ax.axvline(np.mean(dados), color='red', linestyle='--', 
                  label=f'Média: {np.mean(dados):.3f}')
        ax.axvline(1/3, color='green', linestyle=':', 
                  label='Valor equilátero: 1/3')
        ax.set_xlabel(f'Comprimento {lado}')
        ax.set_ylabel('Densidade')
        ax.set_title(f'Distribuição de {lado} em triângulos formados')
        ax.legend()
    
    plt.suptitle('Distribuição dos Lados dos Triângulos Formados', fontsize=14, y=1.05)
    plt.tight_layout()
    return fig

def analise_estatistica(resultados):
    """Análise estatística detalhada dos resultados."""
    p = resultados['p_estimada']
    erro = resultados['erro_padrao']
    triangulos = resultados['triangulos']
    
    print("=" * 60)
    print("ANÁLISE ESTATÍSTICA DOS RESULTADOS")
    print("=" * 60)
    print(f"\nProbabilidade estimada de formar triângulo: {p:.6f}")
    print(f"Erro padrão da estimativa: {erro:.6f}")
    print(f"Intervalo de confiança 95%: ({p-1.96*erro:.6f}, {p+1.96*erro:.6f})")
    print(f"Valor teórico esperado: 0.25")
    print(f"Diferença em relação ao valor teórico: {abs(p-0.25):.6f}")
    
    if len(triangulos) > 0:
        print(f"\nNúmero de triângulos formados: {len(triangulos)}")
        
        # Estatísticas dos triângulos
        print("\nEstatísticas dos triângulos formados:")
        print(f"Média dos lados: a={np.mean(triangulos[:,0]):.4f}, "
              f"b={np.mean(triangulos[:,1]):.4f}, c={np.mean(triangulos[:,2]):.4f}")
        print(f"Desvio padrão dos lados: a={np.std(triangulos[:,0]):.4f}, "
              f"b={np.std(triangulos[:,1]):.4f}, c={np.std(triangulos[:,2]):.4f}")
        
        # Coeficiente de variação médio (medida de equilateridade)
        coef_var = np.mean([np.std(tri) / np.mean(tri) for tri in triangulos])
        print(f"Coeficiente de variação médio: {coef_var:.4f} "
              f"(menor = mais próximo de equilátero)")
        
        # Porcentagem de triângulos "quase equiláteros" (desvio < 10%)
        desvios = [np.std(tri) / np.mean(tri) for tri in triangulos]
        quase_equilateros = sum(d < 0.1 for d in desvios) / len(desvios)
        print(f"Triângulos quase equiláteros (desvio < 10%): {quase_equilateros:.2%}")

# Executar simulação principal
if __name__ == "__main__":
    print("Executando simulação de Monte Carlo com 200,000 iterações...")
    resultados = simulacao_monte_carlo(n_simulacoes=200000)
    
    # Exibir análise estatística
    analise_estatistica(resultados)
    
    # Gerar visualizações
    print("\nGerando visualizações...")
    
    fig1 = plot_histogramas_comprimentos(resultados)
    fig1.savefig('histogramas_comprimentos.png', dpi=150, bbox_inches='tight')
    
    fig2 = plot_diagrama_dispersao_cortes(resultados)
    fig2.savefig('diagrama_dispersao_cortes.png', dpi=150, bbox_inches='tight')
    
    fig3 = plot_espaco_abc(resultados)
    if fig3:
        fig3.savefig('espaco_abc_3d.png', dpi=150, bbox_inches='tight')
    
    fig4 = plot_triangulos_quase_equilateros(resultados)
    if fig4:
        fig4.savefig('triangulos_exemplos.png', dpi=150, bbox_inches='tight')
    
    fig5 = plot_distribuicao_lados_triangulos(resultados)
    if fig5:
        fig5.savefig('distribuicao_lados_triangulos.png', dpi=150, bbox_inches='tight')
    
    print("\nVisualizações salvas como arquivos PNG.")
    print("\nPara exibir os gráficos interativamente, execute:")
    print("plt.show() após cada figura.")
    
    # Mostrar um resumo gráfico
    plt.show()
