import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Necessário para plotar gráficos 3D

# Carrega os dados do arquivo CSV (ignorando a primeira linha de cabeçalho)
data = np.loadtxt('atividade_enzimatica.csv', delimiter=',', skiprows=1)

# Separa as colunas: Temperatura, pH e Atividade Enzimática
temperatura = data[:, 0]
ph = data[:, 1]
atividade_enzimatica = data[:, 2]

# Organiza os dados independentes em uma matriz X e a variável alvo em y
X = np.column_stack((temperatura, ph))
y = atividade_enzimatica

# =====================================
# 2. Normalização dos dados
# =====================================
def normalizar_dados(X):
    """Normaliza as variáveis preditoras para média zero e desvio padrão 1."""
    media = np.mean(X, axis=0)
    desvio = np.std(X, axis=0)
    return (X - media) / desvio

# Normaliza X e adiciona a coluna de 1s (intercepto)
X_norm = normalizar_dados(X)
X_norm = np.c_[np.ones(X_norm.shape[0]), X_norm]

# =====================================
# 3. Definição dos Modelos
# =====================================
def mqo_tradicional(X, y):
    """Implementação de Regressão Linear via MQO (Mínimos Quadrados Ordinários)."""
    return np.linalg.inv(X.T @ X) @ X.T @ y

def mqo_regularizado(X, y, lambda_):
    """Implementação de Regressão Regularizada (Tikhonov/Ridge)."""
    return np.linalg.inv(X.T @ X + lambda_ * np.eye(X.shape[1])) @ X.T @ y

def calcular_rss(y_real, y_pred):
    """Calcula a Soma dos Quadrados dos Resíduos (RSS)."""
    return np.sum((y_real - y_pred) ** 2)

def media_valores_observaveis(y):
    """Retorna a média simples da variável alvo (modelo base)."""
    return np.mean(y)

# =====================================
# 4. Configuração de Validação Monte Carlo
# =====================================
R = 500  # Número de rodadas
lambdas = [0.25, 0.5, 0.75, 1]  # Valores de regularização (λ)

# Armazenamento dos RSS
rss_mqo = []
rss_media = []
rss_reg = {l: [] for l in lambdas}

n = len(X_norm)  # Total de amostras

# =====================================
# 5. Loop de Validação Monte Carlo
# =====================================
for _ in range(R):
    # Gera índices para treino (80%) e teste (20%) de forma aleatória
    idx_treino = np.random.choice(n, int(0.8 * n), replace=False)
    idx_teste = np.setdiff1d(np.arange(n), idx_treino)

    Xtr, Xts = X_norm[idx_treino], X_norm[idx_teste]
    ytr, yts = y[idx_treino], y[idx_teste]

    # MQO Tradicional
    beta = mqo_tradicional(Xtr, ytr)
    ypred = Xts @ beta
    rss_mqo.append(calcular_rss(yts, ypred))

    # MQO Regularizado para diferentes λ
    for l in lambdas:
        beta_l = mqo_regularizado(Xtr, ytr, l)
        ypred_l = Xts @ beta_l
        rss_reg[l].append(calcular_rss(yts, ypred_l))

    # Modelo da Média
    media = media_valores_observaveis(ytr)
    ypred_media = np.full_like(yts, media)
    rss_media.append(calcular_rss(yts, ypred_media))

# =====================================
# 6. Estatísticas dos Resultados
# =====================================
def estatisticas(lista):
    """Calcula média, desvio padrão, máximo e mínimo."""
    return np.mean(lista), np.std(lista), np.max(lista), np.min(lista)

# Exibe os resultados
print("\nTabela de Resultados (RSS):")
print("{:<25} {:>10} {:>10} {:>10} {:>10}".format("Modelo", "Média", "Desvio", "Máximo", "Mínimo"))

# Modelo da Média
media_stats = estatisticas(rss_media)
print("{:<25} {:10.4f} {:10.4f} {:10.4f} {:10.4f}".format("Média variável dependente", *media_stats))

# MQO Tradicional
mqo_stats = estatisticas(rss_mqo)
print("{:<25} {:10.4f} {:10.4f} {:10.4f} {:10.4f}".format("MQO Tradicional", *mqo_stats))

# MQO Regularizado
for l in lambdas:
    stats = estatisticas(rss_reg[l])
    print(f"MQO Regularizado (λ={l:<4}) {stats[0]:10.4f} {stats[1]:10.4f} {stats[2]:10.4f} {stats[3]:10.4f}")

# =====================================
# 7. Gráficos 2D de visualização dos dados
# =====================================

# Atividade Enzimática x pH
plt.figure(figsize=(8, 6))
plt.scatter(ph, y, color='blue', alpha=0.6)
plt.xlabel("pH")
plt.ylabel("Atividade Enzimática")
plt.title("Atividade Enzimática x pH")
plt.grid(True)
plt.show()

# Atividade Enzimática x Temperatura
plt.figure(figsize=(8, 6))
plt.scatter(temperatura, y, color='red', alpha=0.6)
plt.xlabel("Temperatura")
plt.ylabel("Atividade Enzimática")
plt.title("Atividade Enzimática x Temperatura")
plt.grid(True)
plt.show()

# =====================================
# 8. Gráficos 3D comparando os modelos
# =====================================

fig = plt.figure(figsize=(18, 6))

# Gráfico 1: MQO Tradicional
ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(X[:, 0], X[:, 1], y, color='black', alpha=0.6)
ypred_mqo = X_norm @ mqo_tradicional(X_norm, y)
ax1.plot_trisurf(X[:, 0], X[:, 1], ypred_mqo, color='blue', alpha=0.5)
ax1.set_title("MQO Tradicional")
ax1.set_xlabel("Temperatura")
ax1.set_ylabel("pH")
ax1.set_zlabel("Atividade Enzimática")

# Gráfico 2: MQO Regularizado com λ=0.5
ax2 = fig.add_subplot(132, projection='3d')
ax2.scatter(X[:, 0], X[:, 1], y, color='black', alpha=0.6)
ypred_reg = X_norm @ mqo_regularizado(X_norm, y, 0.5)
ax2.plot_trisurf(X[:, 0], X[:, 1], ypred_reg, color='green', alpha=0.5)
ax2.set_title("MQO Regularizado (λ=0.5)")
ax2.set_xlabel("Temperatura")
ax2.set_ylabel("pH")
ax2.set_zlabel("Atividade Enzimática")

# Gráfico 3: Modelo da Média
ax3 = fig.add_subplot(133, projection='3d')
ax3.scatter(X[:, 0], X[:, 1], y, color='black', alpha=0.6)
media = np.mean(y)
ypred_media = np.full_like(y, media)
ax3.plot_trisurf(X[:, 0], X[:, 1], ypred_media, color='red', alpha=0.5)
ax3.set_title("Modelo de Média")
ax3.set_xlabel("Temperatura")
ax3.set_ylabel("pH")
ax3.set_zlabel("Atividade Enzimática")

plt.tight_layout()
plt.show()