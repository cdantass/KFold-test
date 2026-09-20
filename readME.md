O dataset utilizado possui pouco mais de **2 GB**, portanto não foi incluído devido ao seu tamanho.

-------

Para baixá-lo, acesse:
https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd

---------

Após o download, extraia o conteúdo dentro do projeto, depois renomeie a pasta principal para "dataset"

----------

Como executar a atividade(Utilizei o VS Code como IDE):

----------

Abra o terminal(CTRL + J)
1- pip install -r requirements.txt
2- python kfold_comum.py
3- python group_kfold.py
4- python nb.py

----------

Inicialmente comecei analisando apenas o KFold comum, e percebi que a acurácia estava "enganosa", por conta que o mesmo participante aparecia tanto na classe "Drowsy" quanto na "Non Drowsy", e o KFold comum não impedia que imagens da mesma pessoa ficassem em treino e teste ao mesmo tempo, e tivemos um resultado "enganoso":

*Teste com o KFold Comum*
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

Então, resolvi testar o GroupKFold, usando o código do participante como grupo, para garantir que a mesma pessoa não ficasse ao mesmo tempo no treino e no teste, e tive esse resultado:

*Teste com o GroupKFold*
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

*Comparação entre KNN e Naive Bayes*

Ao comparar os resultados obtidos utilizando GroupKFold, foi possível observar que o KNN apresentou a maior acurácia média entre os algoritmos avaliados.

### KNN (GroupKFold)

Accuracy: 63.36%

Desvio Padrão: 12.59%

### Naive Bayes (GroupKFold)

Mean Accuracy: 57.93%

Desvio Padrão: 10.98%

## Análise

Ao observar apenas a acurácia, o KNN obteve o melhor resultado neste conjunto de dados. O melhor modelo encontrado foi o KNN com K = 1, alcançando 63.36% de acerto, enquanto o Naive Bayes atingiu 57.93%.

Em relação à estabilidade, os dois algoritmos apresentaram valores relativamente próximos de desvio padrão. O Naive Bayes foi ligeiramente mais consistente entre os folds, mas essa diferença não foi suficiente para compensar a perda de desempenho observada na acurácia média.

## Sobre custo computacional

Embora o KNN tenha apresentado melhor desempenho neste experimento, ele também possui um custo computacional maior durante a fase de previsão. Isso acontece porque cada nova amostra precisa ser comparada com todas as amostras presentes no conjunto de treinamento para determinar seus vizinhos mais próximos.

O Naive Bayes, por outro lado, possui treinamento e classificação bastante rápidos, sendo uma alternativa interessante quando o objetivo é reduzir tempo de execução ou trabalhar com conjuntos de dados maiores.

### Conclusão

Para o dataset utilizado e para o pré-processamento adotado, o KNN apresentou o melhor desempenho geral, atingindo 63.36% de acurácia com K = 1.

O Naive Bayes apresentou um comportamento mais estável, porém obteve uma acurácia inferior ao KNN.

Resumindo, a melhor escolha nesse projeto foi o algoritmo KNN, por conta do seu desempenho ter sido melhor. Embora o NB tenha apresentado uma estabilidade um pouco maior, essa vantagem foi pequena quando comparada ao ganho de desempenho obtido pelo KNN.

-----
Mas vale lembrar que não existe um algoritmo universalmente *MELHOR*, e sim o mais adequado para tal situação, dependendo do conjunto de dados, pré-processamento e da maneira que os dados serão avaliados ou testados.