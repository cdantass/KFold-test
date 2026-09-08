*KFold e KNN*

O dataset utilizado possui pouco mais de **2 GB**, portanto não foi incluído devido ao seu tamanho.

-------

Para baixá-lo, acesse:
https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd

---------

Após o download, extraia o conteúdo dentro da pasta "dataset"

----------

Como executar a atividade(Utilizei o VS Code como IDE):

----------

Abra o terminal(CTRL + J)

1- pip install -r requirements.txt

2- python kfold_comum.py

3- python group_kfold.py

----------

Inicialmente comecei analisando apenas o KFold comum, e percebi que a acurácia estava "enganosa", por conta que o mesmo participante aparecia tanto na classe "Drowsy" quanto na "Non Drowsy", e o KFold comum não impedia que imagens da mesma pessoa ficassem em treino e teste ao mesmo tempo, e tivemos um resultado "enganoso":

*Teste com o KFold Comum*

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

Então, resolvi testar o GroupKFold, usando o código do participante como grupo, para garantir que a mesma pessoa não ficasse ao mesmo tempo no treino e no teste, e tive esse resultado:

*Teste com o GroupKFold*

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
