# Classificação de expressões com sensores EMG — Código explicativo (Estudo)
# Modelos testados: MQO (regressão), Naive Bayes, Gaussianos (Covariâncias iguais, distintas e regularizadas)

import numpy as np
import matplotlib.pyplot as plt

# =============================================
# 1. Carregamento dos dados e visualização inicial
# =============================================

# Carrega os dados do arquivo CSV sem cabeçalho
data = np.loadtxt('EMGsDataset.csv', delimiter=',')

# As duas primeiras linhas são os sinais dos sensores (features)
# A terceira linha são as classes (valores entre 1 e 5, representando expressões)
X = data[:2, :].T               # Transpõe para deixar cada linha como uma amostra
Y = data[2, :].astype(int)      # Converte os rótulos para inteiros

# ======= AMOSTRAGEM PARA TESTE =======
# Reduz a base para n amostras aleatórias para acelerar a execução, conforme escolha
np.random.seed(42)              # Define semente para reprodutibilidade
idx = np.random.choice(len(X), 25000, replace=False)
X = X[idx]
Y = Y[idx]

# Visualização dos dados em um gráfico de dispersão
colors = ['blue', 'green', 'red', 'orange', 'purple']  # Cores por classe
labels = ['Neutro', 'Sorriso', 'Sobrancelhas Levantadas', 'Surpreso', 'Rabugento']
plt.figure(figsize=(8, 6))
for i in range(1, 6):
    plt.scatter(X[Y == i, 0], X[Y == i, 1], label=labels[i - 1], color=colors[i - 1], s=20, edgecolors='black')
plt.xlabel('Corrugador do Supercílio')
plt.ylabel('Zigomático Maior')
plt.title('Dispersão dos Dados por Classe')
plt.legend()
plt.tight_layout()
plt.show()

# =============================================
# 2. Funções auxiliares
# =============================================

def split(X, Y, p=0.8):
    """Divide os dados em treino e teste aleatoriamente."""
    n = len(X)
    idx = np.random.permutation(n)  # Embaralha os índices
    n_tr = int(p * n)               # Define o tamanho do conjunto de treino
    return X[idx[:n_tr]], Y[idx[:n_tr]], X[idx[n_tr:]], Y[idx[n_tr:]]

def estat(v):
    """Calcula estatísticas básicas: média, desvio padrão, valor máximo e mínimo"""
    return round(np.mean(v), 4), round(np.std(v), 4), round(np.max(v), 4), round(np.min(v), 4)

# =============================================
# 3. Modelos de classificação
# =============================================

# --- MQO como Regressão Linear (regressão tratada como classificação)
def mqo_fit(X, Y):
    Xb = np.c_[np.ones(X.shape[0]), X]  # Adiciona termo de bias (coluna de 1s)
    return np.linalg.pinv(Xb.T @ Xb) @ Xb.T @ Y  # Solução da regressão linear normal

def mqo_predict(X, beta):
    Xb = np.c_[np.ones(X.shape[0]), X]
    return np.clip(np.round(Xb @ beta), 1, 5).astype(int)  # Arredonda e limita entre 1 e 5

# --- Naive Bayes assumindo independência entre as variáveis
def naive_fit(X, Y):
    modelos = []
    for c in range(1, 6):
        Xc = X[Y == c]                     # Filtra amostras da classe c
        mu = Xc.mean(axis=0)              # Média por variável
        var = Xc.var(axis=0) + 1e-6       # Variância por variável (evita divisão por zero)
        logprior = np.log(len(Xc) / len(X))  # Log da probabilidade a priori da classe
        modelos.append((mu, var, logprior))
    return modelos

def naive_predict(X, modelos):
    pred = []
    for x in X:
        scores = []
        for mu, var, logprior in modelos:
            # Cálculo do log da verossimilhança da gaussiana (independente por variável)
            log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * var)) - 0.5 * np.sum(((x - mu) ** 2) / var)
            scores.append(log_likelihood + logprior)
        pred.append(np.argmax(scores) + 1)  # Classe com maior score
    return np.array(pred)

# --- Classificador Gaussiano com covariâncias iguais para todas as classes
def cov_iguais_fit(X, Y):
    mus, priors, pooled_cov = [], [], np.zeros((X.shape[1], X.shape[1]))
    for c in range(1, 6):
        Xc = X[Y == c]
        mus.append(Xc.mean(axis=0))
        priors.append(np.log(len(Xc) / len(X)))
        pooled_cov += (len(Xc) - 1) * np.cov(Xc.T)  # Soma ponderada das covariâncias
    pooled_cov /= (len(X) - 5)
    pooled_cov += 1e-6 * np.eye(X.shape[1])  # Regularização
    return mus, [pooled_cov] * 5, priors

# --- Classificador Gaussiano com covariância individual por classe
def gau_trad_fit(X, Y):
    mus, covs, priors = [], [], []
    for c in range(1, 6):
        Xc = X[Y == c]
        mus.append(Xc.mean(axis=0))
        covs.append(np.cov(Xc.T) + 1e-6 * np.eye(2))  # Regulariza a matriz
        priors.append(np.log(len(Xc) / len(X)))
    return mus, covs, priors

# --- Classificador Gaussiano Regularizado (Shrinkage entre individual e global)
def gau_reg_fit(X, Y, lmbd):
    mus, covs, priors = [], [], []
    cov_agrup = np.cov(X.T)  # Covariância geral
    for c in range(1, 6):
        Xc = X[Y == c]
        nc = len(Xc)
        mu = Xc.mean(axis=0)
        cov = np.cov(Xc.T)
        # Combina covariância de classe com global, controlado por lambda
        reg_cov = ((1 - lmbd) * nc * cov + lmbd * len(X) * cov_agrup) / ((1 - lmbd) * nc + lmbd * len(X))
        mus.append(mu)
        covs.append(reg_cov + 1e-6 * np.eye(2))
        priors.append(np.log(nc / len(X)))
    return mus, covs, priors

# --- Classificador Gaussiano com covariância média entre classes
def gau_agregado_fit(X, Y):
    mus, covs, priors = [], [], []
    for c in range(1, 6):
        Xc = X[Y == c]
        mus.append(Xc.mean(axis=0))
        priors.append(np.log(len(Xc) / len(X)))
        covs.append(np.cov(Xc.T))
    cov_agg = sum(covs) / 5 + 1e-6 * np.eye(2)
    return mus, [cov_agg]*5, priors

# --- Predição Gaussiana genérica (usada por todos os modelos gaussianos)
def gau_predict(X, modelo):
    mus, covs, priors = modelo
    pred = []
    for x in X:
        scores = []
        for mu, cov, lp in zip(mus, covs, priors):
            inv = np.linalg.pinv(cov)          # Inversa da matriz de covariância
            diff = x - mu
            s = -0.5 * diff @ inv @ diff - 0.5 * np.log(np.linalg.det(cov)) + lp
            scores.append(s)
        pred.append(np.argmax(scores) + 1)
    return np.array(pred)

# =============================================
# 4. Avaliação com Monte Carlo (R = 10)
# =============================================

R = 10  # Número de repetições (simulações independentes)
modelos = {
    "MQO": (mqo_fit, mqo_predict),
    "Gaussiano Tradicional": (gau_trad_fit, gau_predict),
    "Gaussiano (Cov. de todo cj. treino)": (cov_iguais_fit, gau_predict),
    "Classificador Gaussiano (Cov. Agregada)": (gau_agregado_fit, gau_predict),
    "Classificador de Bayes Ingênuo (Naive Bayes Classifier)": (naive_fit, naive_predict),
    "Regularizado λ=0.25": (lambda X, Y: gau_reg_fit(X, Y, 0.25), gau_predict),
    "Regularizado λ=0.5": (lambda X, Y: gau_reg_fit(X, Y, 0.5), gau_predict),
    "Regularizado λ=0.75": (lambda X, Y: gau_reg_fit(X, Y, 0.75), gau_predict),
}

# Inicializa dicionário com resultados
resultados = {nome: [] for nome in modelos}

# R rodadas de avaliação com conjuntos de treino/teste diferentes (Monte Carlo)
for _ in range(R):
    Xtr, Ytr, Xts, Yts = split(X, Y)
    for nome, (fit, pred) in modelos.items():
        modelo = fit(Xtr, Ytr)
        Ypred = pred(Xts, modelo)
        acc = np.mean(Ypred == Yts)  # Acurácia
        resultados[nome].append(acc)

# =============================================
# 5. Impressão da Tabela Final
# =============================================

# Cabeçalho da tabela
print("\nTabela Final de Desempenho:")
print("Modelo\t\t\t\t\t\t\tMédia\t\tDesvio\t\tMínimo\t\tMáximo")
#print("{:<25} {:>8} {:>15} {:>15} {:>15}".format("Modelo", "Média", "Desvio-Padrão", "Maior Valor", "Menor Valor"))

# Exibe estatísticas de cada modelo
for nome in modelos:
    media, desvio, maior, menor = estat(resultados[nome])
    print(f"{nome:50s}\t{media:.4f}\t\t{desvio:.4f}\t\t{menor:.4f}\t\t{maior:.4f}")
    #print("{:<25} {:>8} {:>15} {:>15} {:>15}".format(nome, media, desvio, maior, menor))

# Salva em um arquivo CSV simples (pode ser aberto no Excel)
with open("tabela_resultados.csv", "w", encoding="utf-8") as f:
    f.write("Modelo,Média,Desvio-Padrão,Maior Valor,Menor Valor\n")
    for nome in modelos:
        media, desvio, maior, menor = estat(resultados[nome])
        f.write(f"{nome},{media},{desvio},{maior},{menor}\n")


