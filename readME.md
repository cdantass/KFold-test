# Projetos de Machine Learning

Este repositório reúne projetos desenvolvidos para aplicar e comparar diferentes técnicas de Machine Learning. Cada projeto está organizado em sua própria pasta, com os códigos, datasets utilizados e resultados gerados.

## Projetos

### Algoritmo SVM

Aplicação do Support Vector Machine em um problema de classificação. O projeto utiliza o SVM para encontrar uma separação entre as classes presentes no dataset.

### Apriori

Aplicação do algoritmo Apriori em um dataset transacional de compras. O projeto identifica conjuntos de itens frequentes e gera regras de associação utilizando suporte, confiança e lift.

O suporte mínimo escolhido foi de 5%, pois apresentou um equilíbrio entre a quantidade de resultados e a relevância das combinações encontradas.

### Clusterização

Aplicação dos algoritmos K-Means e DBSCAN para identificar grupos no dataset sem utilizar classes durante o treinamento.

O projeto também apresenta métricas e visualizações para facilitar a análise dos clusters encontrados.

### KFold

Comparação entre KFold comum e GroupKFold em um problema de classificação de imagens.

O KFold comum apresentou uma acurácia muito alta porque imagens dos mesmos participantes poderiam aparecer no treino e no teste. O GroupKFold foi utilizado para separar os participantes entre os grupos e produzir uma avaliação mais próxima de um cenário real.

### Previsão de séries

Aplicação dos modelos ARIMA e LSTM para previsão de séries temporais.

Os resultados de cada modelo são salvos em gráficos para permitir a comparação entre os valores reais e os valores previstos.

### Random Forest

Aplicação do algoritmo Random Forest em um problema de classificação. O projeto compara seu desempenho com outros algoritmos utilizando métricas de acurácia e desvio padrão.

## Estrutura

```text
KFold-test/
├── Algoritmo SVM/
├── Apriori/
├── clusterização/
├── KFold/
├── Previsão de séries/
├── Random Forest/
├── .gitignore
├── README.md
└── requirements.txt