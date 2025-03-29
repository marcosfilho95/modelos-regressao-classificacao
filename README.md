# Trabalho Computacional AV1: Modelos de Regressão e Classificação

**Desenvolvedores:**  
- Marcos Antonio Felix – 1810449  
- Gil Melo Bandeira Torres – 1720537  

---

## 1. Descrição do Projeto

Este projeto, desenvolvido no âmbito da disciplina de *Inteligência Artificial Computacional*, apresenta uma análise comparativa entre diferentes modelos estatísticos aplicados a dois problemas:

### A. Tarefa de Regressão

- **Objetivo:**  
  Prever a atividade enzimática com base em variáveis preditoras, especificamente **pH** e **temperatura**.

- **Dados:**  
  Utiliza o arquivo `atividade_enzimatica.csv`, onde cada amostra possui medições de temperatura, pH e o nível de atividade enzimática.

- **Modelos Implementados:**
  - **MQO Tradicional:** modelo clássico que minimiza a soma dos quadrados dos resíduos (RSS).
  - **MQO Regularizado (Ridge/Tikhonov):** incorpora um termo de penalização (λ) para reduzir overfitting.
  - **Modelo da Média Simples:** utiliza a média dos valores observáveis como linha de base.

- **Metodologia:**  
  Os dados foram organizados em uma matriz de variáveis independentes (X) e um vetor de resposta (y).  
  A validação foi realizada por meio de **simulações de Monte Carlo (500 iterações)**, onde os dados foram particionados aleatoriamente (80% treino, 20% teste) e os modelos foram avaliados pelo RSS.

- **Conclusão:**  
  Os modelos de MQO tradicional e regularizado apresentaram desempenho similar, com o modelo da média se mostrando inadequado para capturar a variação da atividade enzimática.

---

### B. Tarefa de Classificação

- **Objetivo:**  
  Desenvolver um sistema para categorizar sinais de eletromiografia (EMG) captados por dois sensores – **Corrugador do Supercílio** e **Zigomático Maior** – em cinco expressões faciais.

- **Classes e Codificação:**

Neutro: [ 1, -1, -1, -1, -1 ] <br>
Sorriso: [-1, 1, -1, -1, -1 ] <br>
Sobrancelhas Levantadas: [-1, -1, 1, -1, -1 ] <br>
Surpreso: [-1, -1, -1, 1, -1 ] <br>
Rabugento: [-1, -1, -1, -1, 1 ]


- **Modelos Implementados:**
- **MQO Adaptado para Classificação:** modelo de regressão linear arredondado para categorizar.
- **Classificador Bayes Ingênuo (Naive Bayes):** baseado na hipótese de independência condicional entre as variáveis.
- **Classificadores Gaussianos:**
  - **CG Tradicional:** estima média e matriz de covariância para cada classe.
  - **CG com Cov. Igual (Global):** assume uma matriz de covariância comum a todas as classes.
  - **CG com Cov. Agregada:** utiliza a média das matrizes de covariância das classes.
  - **CG Regularizado (Friedman):** aplica regularização (testada para λ = 0.25, 0.5, 0.75) para mitigar overfitting.

- **Metodologia:**  
Os dados foram organizados em uma matriz de características (X) e um vetor de rótulos (Y).  
Uma visualização inicial (gráfico de dispersão) evidenciou agrupamentos e sobreposição entre as classes.  
A validação dos modelos foi realizada via **simulações de Monte Carlo (500 iterações)**, onde a **acurácia** – definida como o número de previsões corretas dividido pelo número total de amostras de teste – foi utilizada para mensurar a performance.

- **Conclusão:**  
Os classificadores Gaussianos (CG Tradicional, CG Cov. Global e CG Cov. Agregada) alcançaram acurácias superiores a **96%** com baixa variabilidade, demonstrando robustez e alta capacidade de generalização.  
Em contraste, o MQO Adaptado e o Naive Bayes apresentaram desempenho inferior, confirmando que a modelagem probabilística (com uso de médias e covariâncias) é mais eficaz para a classificação dos sinais EMG.
