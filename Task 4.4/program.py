import networkx as nx
import matplotlib.pyplot as plt

G = nx.karate_club_graph()

print("--- Інформація про граф 'Karate Club' ---")
print(f"Кількість вузлів (учасників): {G.number_of_nodes()}")
print(f"Кількість ребер (зв'язків): {G.number_of_edges()}")
print("------------------------------------------\n")

print("### 1. Аналіз за допомогою Пошуку в ширину (BFS) ###")

try:
    path = nx.shortest_path(G, source=0, target=33)
    print(f"Найкоротший шлях між вузлом 0 та 33: {path}")
    print(f"Довжина шляху (кількість 'стрибків'): {len(path) - 1}")
except nx.NetworkXNoPath:
    print("Шляху між вузлами 0 та 33 не існує.")

dist_2_nodes = nx.descendants_at_distance(G, source=0, distance=2)
print(f"\nВузли на відстані 2 'стрибків' від вузла 0: {dist_2_nodes}")

print("\n### 2. Аналіз за допомогою Пошуку в глибину (DFS) ###")

reachable_nodes = nx.descendants(G, source=0)
print(f"Всі вузли, досяжні з вузла 0: {len(reachable_nodes)} вузлів")

dfs_tree = nx.dfs_tree(G, source=0)
print(f"Кількість ребер у DFS-дереві: {dfs_tree.number_of_edges()}")

print("\n### 3. 'Ground-Truth' Спільноти (Як у Friendster) ###")

club_communities = nx.get_node_attributes(G, 'club')

community_A = [node for node, club in club_communities.items() if club == 'Mr. Hi']
community_B = [node for node, club in club_communities.items() if club == 'Officer']

print(f"Спільнота 'Mr. Hi': {len(community_A)} учасників")
print(f"Спільнота 'Officer': {len(community_B)} учасників")

print("\nСтворення візуалізації графа...")

color_map = []
for node in G:
    if club_communities[node] == 'Mr. Hi':
        color_map.append('#1f78b4')
    else:
        color_map.append('#ff7f00')

plt.figure(figsize=(10, 8))
nx.draw(G, 
        node_color=color_map, 
        with_labels=True, 
        node_size=600, 
        font_weight='bold')
plt.title("Візуалізація 'Karate Club' з Ground-Truth Спільнотами")
plt.show()

print("Візуалізацію завершено.")