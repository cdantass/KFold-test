# Apriori em dataset transacional de cafeteria

Este projeto aplica o algoritmo Apriori ao arquivo `dataset_transacional_de_compras_apriori.txt`, que contém **999 transações** e **59 produtos distintos**.

## Estrutura esperada

Coloque os arquivos na mesma pasta `ML`:

```text
ML/
├── apriori.py
└── dataset_transacional_de_compras_apriori.txt
```

## Como executar

```bash
cd ML
python apriori.py
```

O programa gera:

- `itemsets_frequentes.csv`
- `regras_associacao.csv`

## MinSup escolhido: 5%

Foi adotado `minsup = 0.05`, equivalente a exigir que um itemset apareça em pelo menos **50 das 999 transações**.

A escolha foi feita comparando diferentes valores:

| MinSup | Itemsets frequentes | Pares frequentes |
|---:|---:|---:|
| 1% | 78 | 17 |
| 3% | 45 | 6 |
| **5%** | **19** | **5** |
| 8% | 12 | 4 |
| 10% | 11 | 3 |

Com 1%, surgem combinações presentes em apenas cerca de 10 compras, aumentando o risco de regras ocasionais. Valores de 8% ou 10% eliminam padrões relevantes. O valor de 5% oferece o melhor equilíbrio: reduz associações raras e preserva cinco combinações fortes e interpretáveis.

## Principais pares encontrados com MinSup de 5%

| Combinação | Ocorrências | Suporte |
|---|---:|---:|
| Café Expresso + Pão de Queijo | 246 | 24,62% |
| Cappuccino + Croissant de Manteiga | 133 | 13,31% |
| Bolo de Cenoura + Café Coado | 102 | 10,21% |
| Sanduíche Natural + Suco de Laranja | 84 | 8,41% |
| Cookie de Chocolate + Iced Coffee | 52 | 5,21% |

Além do suporte mínimo, as regras são filtradas com **confiança mínima de 70%**. O `lift` também é calculado para verificar se a associação é superior ao que seria esperado pela frequência individual dos produtos. Um `lift > 1` indica associação positiva.

## Decisão

O `minsup = 5%` é adequado porque mantém padrões recorrentes do comportamento de compra, produz uma quantidade pequena e analisável de resultados e evita que associações muito raras sejam tratadas como relevantes.
