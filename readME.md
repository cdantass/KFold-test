<<<<<<< HEAD
=======
*KFold e KNN*

>>>>>>> origin/main
O dataset utilizado possui pouco mais de **2 GB**, portanto não foi incluído devido ao seu tamanho.

-------

Para baixá-lo, acesse:
https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd

---------

<<<<<<< HEAD
Após o download, extraia o conteúdo dentro do projeto, depois renomeie a pasta principal para "dataset"
=======
Após o download, extraia o conteúdo dentro da pasta "dataset"
>>>>>>> origin/main

----------

Como executar a atividade(Utilizei o VS Code como IDE):

----------

Abra o terminal(CTRL + J)
<<<<<<< HEAD
1- pip install -r requirements.txt
2- python kfold_comum.py
3- python group_kfold.py
4- python nb.py
=======

1- pip install -r requirements.txt

2- python kfold_comum.py

3- python group_kfold.py
>>>>>>> origin/main

----------

Inicialmente comecei analisando apenas o KFold comum, e percebi que a acurácia estava "enganosa", por conta que o mesmo participante aparecia tanto na classe "Drowsy" quanto na "Non Drowsy", e o KFold comum não impedia que imagens da mesma pessoa ficassem em treino e teste ao mesmo tempo, e tivemos um resultado "enganoso":

*Teste com o KFold Comum*

<<<<<<< HEAD
X format:  (20000, 1024)

Y format:  (20000,)

K=1 | Accuracy: 99.91% | Std: 0.06%

K=2 | Accuracy: 99.91% | Std: 0.05%

K=3 | Accuracy: 99.88% | Std: 0.06%

K=4 | Accuracy: 99.88% | Std: 0.07%

K=5 | Accuracy: 99.86% | Std: 0.07%

K=6 | Accuracy: 99.85% | Std: 0.07%

K=7 | Accuracy: 99.86% | Std: 0.07%

K=8 | Accuracy: 99.83% | Std: 0.07%

K=9 | Accuracy: 99.84% | Std: 0.07%

K=10 | Accuracy: 99.82% | Std: 0.08%

Best K: 1 | Best Accuracy: 99.91% | Std: 0.06%
=======
X format: (20000, 1024)

Y format: (20000,)

K=1 | Accuracy: 99.91%

K=2 | Accuracy: 99.91%

K=3 | Accuracy: 99.88%

K=4 | Accuracy: 99.88%

K=5 | Accuracy: 99.86%

K=6 | Accuracy: 99.85%

K=7 | Accuracy: 99.86%

K=8 | Accuracy: 99.83%

K=9 | Accuracy: 99.84%

K=10 | Accuracy: 99.82%

K=11 | Accuracy: 99.83%

K=12 | Accuracy: 99.83%

K=13 | Accuracy: 99.83%

K=14 | Accuracy: 99.82%

K=15 | Accuracy: 99.83%

K=16 | Accuracy: 99.81%

K=17 | Accuracy: 99.81%

K=18 | Accuracy: 99.80%

K=19 | Accuracy: 99.80%

K=20 | Accuracy: 99.79%

Melhor K: 1 | Melhor Acurácia: 99.91%
>>>>>>> origin/main

Então, resolvi testar o GroupKFold, usando o código do participante como grupo, para garantir que a mesma pessoa não ficasse ao mesmo tempo no treino e no teste, e tive esse resultado:

*Teste com o GroupKFold*

<<<<<<< HEAD
X format:  (20000, 1024)

Y format:  (20000,)

K=1 | Accuracy: 63.36% | Std: 12.59%

K=2 | Accuracy: 62.99% | Std: 12.97%

K=3 | Accuracy: 62.05% | Std: 12.82%

K=4 | Accuracy: 62.23% | Std: 12.99%

K=5 | Accuracy: 61.75% | Std: 12.59%

K=6 | Accuracy: 61.70% | Std: 12.72%

K=7 | Accuracy: 61.84% | Std: 11.90%

K=8 | Accuracy: 61.95% | Std: 12.04%

K=9 | Accuracy: 60.71% | Std: 10.93%

K=10 | Accuracy: 60.66% | Std: 11.03%

Best K: 1 | Best Accuracy: 63.36% | Std: 12.59%



*Teste com o Naive Bayes*

Fold 1: 48.98%

Fold 2: 48.42%

Fold 3: 78.46%

Fold 4: 54.86%

Fold 5: 58.91%

Mean Accuracy: 57.93%

Standard Deviation: 10.98%

----------

*Comparação entre os 4 algoritmos (GroupKFold)*

X format:  (20000, 1024)

Y format:  (20000,)

Participantes únicos:  28

Decision Tree | Accuracy: 52.22% | Std: 7.34%

Random Forest | Accuracy: 52.75% | Std: 11.57%

Naive Bayes | Accuracy: 54.09% | Std: 13.93%

KNN (k=1) | Accuracy: 62.67% | Std: 10.26%

KNN (k=2) | Accuracy: 62.68% | Std: 10.77%

KNN (k=3) | Accuracy: 62.15% | Std: 10.69%

KNN (k=4) | Accuracy: 62.26% | Std: 10.96%

KNN (k=5) | Accuracy: 61.89% | Std: 11.04%

KNN (k=6) | Accuracy: 61.87% | Std: 11.29%

KNN (k=7) | Accuracy: 62.00% | Std: 11.70%

KNN (k=8) | Accuracy: 62.11% | Std: 11.82%

KNN (k=9) | Accuracy: 61.70% | Std: 11.76%

KNN (k=10) | Accuracy: 61.86% | Std: 11.79%

Melhor algoritmo: KNN (k=2) | Accuracy: 62.68% | Std: 10.77%

----------

## Análise

O KNN foi o algoritmo com a melhor acurácia média, chegando a 62.68% com K = 2. Os outros três ficaram bem abaixo: Naive Bayes com 54.09%, Random Forest com 52.75% e Decision Tree com 52.22%. A diferença do KNN para o segundo colocado foi de cerca de 8 pontos percentuais, a maior diferença entre os algoritmos testados.

Dentro do próprio KNN, o valor de K quase não mudou o resultado. Do K=1 ao K=10 a acurácia ficou entre 61.70% e 62.68%, então escolher entre um K e outro não faz muita diferença nesse dataset.

Decision Tree, Random Forest e Naive Bayes ficaram bem próximos entre si, e perto de 50%, que seria o acerto de um chute em um problema com duas classes. Ou seja, com imagens 32x32 em escala de cinza e pixels como atributos, esses modelos tiveram bastante dificuldade para generalizar para participantes que nunca viram no treino.
=======
X format: (20000, 1024)

Y format: (20000,)

K=1 | Accuracy: 61.49%

K=2 | Accuracy: 57.58%

K=3 | Accuracy: 58.88%

K=4 | Accuracy: 57.80%

K=5 | Accuracy: 58.51%

K=6 | Accuracy: 58.35%

K=7 | Accuracy: 58.90%

K=8 | Accuracy: 58.54%

K=9 | Accuracy: 58.94%

K=10 | Accuracy: 58.70%

K=11 | Accuracy: 58.87%

K=12 | Accuracy: 58.67%

K=13 | Accuracy: 58.91%

K=14 | Accuracy: 58.64%

K=15 | Accuracy: 58.69%

K=16 | Accuracy: 58.48%

K=17 | Accuracy: 58.78%

K=18 | Accuracy: 58.56%

K=19 | Accuracy: 58.79%

K=20 | Accuracy: 58.58%

Melhor K: 1 | Melhor Acurácia: 61.49%
>>>>>>> origin/main
