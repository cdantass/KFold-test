# 🤖 Projetos de Machine Learning

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Em%20desenvolvimento-2EA44F?style=for-the-badge)

Este repositório reúne projetos desenvolvidos para aplicar e comparar diferentes técnicas de **Machine Learning**.

Cada projeto está organizado em sua própria pasta, contendo os códigos, datasets utilizados, resultados e instruções de execução.

---

## 📚 Projetos

### 📈 Algoritmo SVM

Aplicação do **Support Vector Machine** em um problema de classificação.

O projeto utiliza o SVM para encontrar uma separação entre as classes presentes no dataset.

---

### 🛒 Apriori

Aplicação do algoritmo **Apriori** em um dataset transacional de compras.

O projeto identifica conjuntos de itens frequentes e gera regras de associação utilizando:

- Suporte
- Confiança
- Lift

O suporte mínimo escolhido foi de **5%**, pois apresentou um equilíbrio entre a quantidade de resultados e a relevância das combinações encontradas.

---

### 🔍 Clusterização

Aplicação dos algoritmos **K-Means** e **DBSCAN** para identificar grupos e padrões sem utilizar classes durante o treinamento.

O projeto apresenta métricas, gráficos e visualizações para facilitar a análise dos clusters encontrados.

---

### 🔄 KFold

Comparação entre **KFold** e **GroupKFold** em um problema de classificação de imagens.

O KFold comum apresentou uma acurácia muito alta porque imagens dos mesmos participantes poderiam aparecer no treino e no teste.

O GroupKFold foi utilizado para separar os participantes entre os grupos e produzir uma avaliação mais próxima de um cenário real.

---

### ⏳ Previsão de séries

Aplicação dos modelos **ARIMA** e **LSTM** para previsão de séries temporais.

Os resultados de cada modelo são apresentados em gráficos, permitindo comparar os valores reais com os valores previstos.

---

### 🌳 Random Forest

Aplicação do algoritmo **Random Forest** em um problema de classificação.

O projeto analisa o desempenho do algoritmo utilizando métricas como acurácia média e desvio padrão.

---

## 📁 Estrutura

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
```

Cada pasta contém os arquivos relacionados ao projeto correspondente.

As informações específicas sobre datasets, parâmetros, resultados e execução estão disponíveis no `README.md` de cada projeto.

---

## 🛠️ Tecnologias utilizadas

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Statsmodels](https://img.shields.io/badge/Statsmodels-4051B5?style=flat-square&logo=python&logoColor=white)

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- OpenCV
- TensorFlow
- Statsmodels

---

## ⚙️ Instalação

Clone o repositório:

```bash
git clone https://github.com/cdantass/KFold-test.git
```

Entre na pasta:

```bash
cd KFold-test
```

Instale as dependências gerais:

```bash
pip install -r requirements.txt
```

Alguns projetos possuem um arquivo `requirements.txt` próprio. Nesse caso, entre na pasta correspondente e instale as dependências:

```bash
cd "nome-do-projeto"
pip install -r requirements.txt
```

---

## ▶️ Execução

Entre na pasta do projeto desejado:

```bash
cd "nome-do-projeto"
```

Execute o arquivo principal informado no README da pasta:

```bash
python nome_do_arquivo.py
```

---

## 🗂️ Datasets

Alguns datasets não foram incluídos no repositório devido ao tamanho dos arquivos.

Quando necessário, o README de cada projeto apresenta o link para download e as instruções para organizar o dataset.

---

## 🎯 Objetivo

O objetivo deste repositório é registrar a aplicação prática de:

- Algoritmos supervisionados
- Algoritmos não supervisionados
- Métodos de validação cruzada
- Previsão de séries temporais
- Regras de associação
- Análise e comparação de modelos
