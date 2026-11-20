import csv
from collections import Counter
from itertools import combinations
from pathlib import Path

DATASET = Path(__file__).parent / "dataset_transacional_de_compras_apriori.txt"
MIN_SUPPORT = 0.05
MIN_CONFIDENCE = 0.70

def load_transactions(path):
    transactions = []

    with path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.reader(file, skipinitialspace=True):
            items = [item.strip() for item in row[1:] if item.strip()]

            if items:
                transactions.append(frozenset(items))

    if not transactions:
        raise ValueError("Nenhuma transação foi encontrada.")

    return transactions

def apriori(transactions, min_support):
    total = len(transactions)
    item_counts = Counter(
        item for transaction in transactions for item in transaction)

    current = {
        frozenset([item]): count / total
        for item, count in item_counts.items()
        if count / total >= min_support
    }

    frequent = dict(current)
    size = 2

    while current:
        previous = list(current)
        candidates = set()

        for index, left in enumerate(previous):
            for right in previous[index + 1:]:
                candidate = left | right

                if len(candidate) == size and all(
                    frozenset(part) in current
                    for part in combinations(candidate, size - 1)
                ):
                    candidates.add(candidate)

        counts = Counter()

        for transaction in transactions:
            for candidate in candidates:
                if candidate <= transaction:
                    counts[candidate] += 1

        current = {
            candidate: count / total
            for candidate, count in counts.items()
            if count / total >= min_support
        }

        frequent.update(current)
        size += 1

    return frequent

def generate_rules(frequent, min_confidence):
    rules = []

    for itemset, support in frequent.items():
        for size in range(1, len(itemset)):
            for part in combinations(itemset, size):
                antecedent = frozenset(part)
                consequent = itemset - antecedent
                confidence = support / frequent[antecedent]
                lift = confidence / frequent[consequent]

                if confidence >= min_confidence:
                    rules.append(
                        (antecedent, consequent, support, confidence, lift))

    return sorted(rules, key=lambda rule: rule[4], reverse=True)

def format_items(itemset):
    return " + ".join(sorted(itemset))

def save_results(frequent, rules):
    with open("itemsets_frequentes.csv", "w",
              encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["itemset", "quantidade_itens", "suporte"])

        for itemset, support in sorted(
            frequent.items(), key=lambda item: item[1], reverse=True):
            writer.writerow([
                format_items(itemset),
                len(itemset),
                f"{support:.6f}"])

    with open("regras_associacao.csv", "w",
              encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "antecedente", "consequente", "suporte", "confianca", "lift"])

        for antecedent, consequent, support, confidence, lift in rules:
            writer.writerow([format_items(antecedent),format_items(consequent),
                f"{support:.6f}",
                f"{confidence:.6f}",
                f"{lift:.6f}"
            ])

def main():
    transactions = load_transactions(DATASET)
    products = set().union(*transactions)

    print("Informações do dataset")
    print(f"Compras: {len(transactions)}")
    print(f"Produtos: {len(products)}")
    print("\nComparação do suporte mínimo")

    for value in [0.01, 0.03, 0.05, 0.08, 0.10]:
        result = apriori(transactions, value)
        pairs = sum(len(itemset) == 2 for itemset in result)
        print(f"{value:.0%}: {len(result)} conjuntos e {pairs} pares")

    frequent = apriori(transactions, MIN_SUPPORT)
    rules = generate_rules(frequent, MIN_CONFIDENCE)
    save_results(frequent, rules)

    print("\nResultado")
    print(f"Suporte mínimo: {MIN_SUPPORT:.0%}")
    print(f"Confiança mínima: {MIN_CONFIDENCE:.0%}")
    print(f"Conjuntos encontrados: {len(frequent)}")
    print(f"Regras encontradas: {len(rules)}")
    print("\nPrincipais regras")

    for antecedent, consequent, support, confidence, lift in rules[:10]:
        print(f"\n{format_items(antecedent)} -> {format_items(consequent)}")
        print(
            f"Suporte: {support:.2%} | "
            f"Confiança: {confidence:.2%} | Lift: {lift:.2f}"
        )
    print("\nOs resultados foram salvos nos arquivos CSV.")
if __name__ == "__main__":
    main()