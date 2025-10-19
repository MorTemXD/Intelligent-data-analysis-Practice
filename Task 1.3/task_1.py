import itertools
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

def generate_candidates(itemsets, k):
    """
    Генерує кандидатів розміру k з частих наборів розміру k-1.
    """
    candidates = set()
    for i in range(len(itemsets)):
        for j in range(i + 1, len(itemsets)):
            set1 = itemsets[i]
            set2 = itemsets[j]
            union = set1.union(set2)
            if len(union) == k:
                candidates.add(union)
    return list(candidates)

def count_support(transactions, candidates):
    """
    Підраховує підтримку для кожного кандидата в транзакціях.
    """
    counts = defaultdict(int)
    for candidate in candidates:
        candidate_frozenset = frozenset(candidate)
        for transaction in transactions:
            if candidate_frozenset.issubset(transaction):
                counts[candidate_frozenset] += 1
    return counts

def apriori(transactions, min_support):
    """
    Алгоритм Apriori для знаходження частих наборів елементів.
    
    Параметри:
    - transactions: список множин, кожна множина — це транзакція елементів.
    - min_support: float, мінімальний поріг підтримки (від 0 до 1).
    
    Повертає:
    - freq_itemsets: словник, де ключ — розмір набору, значення — список частих наборів (frozensets).
    """
    transactions = [set(t) for t in transactions]
    num_transactions = len(transactions)
    abs_min_support = min_support * num_transactions
    
    all_items = set()
    for t in transactions:
        all_items.update(t)
    
    candidates = [{item} for item in all_items]
    freq_itemsets = defaultdict(list)
    
    k = 1
    while candidates:
        counts = count_support(transactions, candidates)
        current_freq = [itemset for itemset, count in counts.items() if count >= abs_min_support]
        if current_freq:
            freq_itemsets[k] = [frozenset(s) for s in current_freq]
            k += 1
            candidates = generate_candidates(current_freq, k)
        else:
            break
    
    return freq_itemsets

def generate_rules(freq_itemsets, transactions, min_confidence):
    """
    Генерує асоціативні правила з частих наборів елементів.
    
    Параметри:
    - freq_itemsets: словник з apriori.
    - transactions: список множин.
    - min_confidence: float, мінімальний поріг довіри (від 0 до 1).
    
    Повертає:
    - rules: список кортежів (антецедент, консеквент, підтримка, довіра).
    - support_dict: словник підтримки для всіх частих наборів.
    """
    num_transactions = len(transactions)
    rules = []
    
    support_dict = {}
    for level in freq_itemsets.values():
        for itemset in level:
            count = sum(1 for t in transactions if itemset.issubset(t))
            support_dict[itemset] = count / num_transactions
    
    for k in range(2, max(freq_itemsets.keys()) + 1):
        for itemset in freq_itemsets[k]:
            for antecedent_size in range(1, k):
                for antecedent in itertools.combinations(itemset, antecedent_size):
                    antecedent = frozenset(antecedent)
                    consequent = itemset - antecedent
                    if len(consequent) == 0:
                        continue
                    conf = support_dict[itemset] / support_dict[antecedent]
                    if conf >= min_confidence:
                        supp = support_dict[itemset]
                        rules.append((antecedent, consequent, supp, conf))
    
    return rules, support_dict

# Основна частина: зчитування з CSV, виконання і створення графіків
if __name__ == "__main__":
    # Зчитуємо CSV-файл
    # Замініть 'transactions.csv' на шлях до вашого CSV-файлу
    df = pd.read_csv('transactions.csv')
    
    # Перетворюємо рядки DataFrame у список множин (транзакцій)
    transactions = [set(df.columns[df.iloc[i] == 1]) for i in range(len(df))]
    
    # Встановлюємо параметри
    min_support = 0.6  # 60% мінімальна підтримка
    min_confidence = 0.7  # 70% мінімальна довіра
    
    # Виконуємо алгоритм Apriori
    freq_itemsets = apriori(transactions, min_support)
    print("Часті набори елементів:")
    for k, itemsets in freq_itemsets.items():
        print(f"Розмір {k}: {itemsets}")
    
    # Генеруємо асоціативні правила та отримуємо підтримку
    rules, support_dict = generate_rules(freq_itemsets, transactions, min_confidence)
    print("\nАсоціативні правила:")
    for antecedent, consequent, supp, conf in rules:
        print(f"{antecedent} => {consequent} (підтримка: {supp:.2f}, довіра: {conf:.2f})")
    
    # Графік 1: Гістограма для частих наборів елементів
    itemset_labels = [', '.join(itemset) for k in freq_itemsets for itemset in freq_itemsets[k]]
    itemset_supports = [support_dict[itemset] for k in freq_itemsets for itemset in freq_itemsets[k]]
    
    if itemset_labels:  # Перевіряємо, чи є часті набори
        plt.figure(figsize=(10, 5))
        plt.bar(itemset_labels, itemset_supports, color='#36A2EB')
        plt.xlabel('Набори елементів')
        plt.ylabel('Підтримка')
        plt.title('Підтримка частих наборів елементів')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    
    # Графік 2: Гістограма для асоціативних правил
    rule_labels = [f"{', '.join(antecedent)} => {', '.join(consequent)}" for antecedent, consequent, _, _ in rules]
    rule_supports = [supp for _, _, supp, _ in rules]
    rule_confidences = [conf for _, _, _, conf in rules]
    
    if rule_labels:  # Перевіряємо, чи є правила
        plt.figure(figsize=(10, 5))
        x = range(len(rule_labels))
        plt.bar([i - 0.2 for i in x], rule_supports, width=0.4, label='Підтримка', color='#36A2EB')
        plt.bar([i + 0.2 for i in x], rule_confidences, width=0.4, label='Довіра', color='#FF6384')
        plt.xlabel('Асоціативні правила')
        plt.ylabel('Значення')
        plt.title('Підтримка та довіра асоціативних правил')
        plt.xticks(x, rule_labels, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.show()