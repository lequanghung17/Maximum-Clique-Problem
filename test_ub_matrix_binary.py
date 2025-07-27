import os
import time
import networkx as nx
from scipy.linalg import eigvals
from pysat.solvers import Glucose3
from pysat.formula import CNF
from pysat.card import CardEnc
import numpy as np

def read_clq_file(filename):
    edges = []
    n = None
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('p'):
                parts = line.strip().split()
                n = int(parts[2])
            elif line.startswith('e'):
                _, u, v = line.strip().split()
                edges.append((int(u), int(v)))
    return n, edges

def greedy_clique(n, edges):
    # Tìm clique lớn nhất bằng greedy cho cận dưới
    adj_sets = [set() for _ in range(n+1)]
    for u, v in edges:
        adj_sets[u].add(v)
        adj_sets[v].add(u)
    best = []
    for start in range(1, n+1):
        clique = [start]
        for v in range(1, n+1):
            if v not in clique and all(v in adj_sets[u] for u in clique):
                clique.append(v)
        if len(clique) > len(best):
            best = clique
    return len(best)

# def prune_by_degree(n, edges, lower_bound):
#     degree = [0] * (n + 1)
#     for u, v in edges:
#         degree[u] += 1
#         degree[v] += 1
#     # Giữ lại các đỉnh bậc đủ lớn
#     valid_nodes = set(i for i in range(1, n+1) if degree[i] >= lower_bound - 1)
#     # Map lại node index cho subgraph
#     old2new = {old: idx+1 for idx, old in enumerate(sorted(valid_nodes))}
#     new_edges = [(old2new[u], old2new[v]) for u, v in edges if u in valid_nodes and v in valid_nodes]
#     return len(valid_nodes), new_edges, old2new, {v for v in valid_nodes}

# def top_degree_subgraph(n, edges, top_k=100):
#     degree = [0] * (n + 1)
#     for u, v in edges:
#         degree[u] += 1
#         degree[v] += 1
#     sorted_nodes = sorted(range(1, n+1), key=lambda i: degree[i], reverse=True)
#     selected = set(sorted_nodes[:top_k])
#     # Remap node index để dễ MCP
#     old2new = {old: idx+1 for idx, old in enumerate(sorted(selected))}
#     new_edges = [(old2new[u], old2new[v]) for u, v in edges if u in selected and v in selected]
#     return len(selected), new_edges, old2new, selected


# def greedy_coloring_sets(n, edges):
#     adj_sets = [set() for _ in range(n+1)]
#     for u, v in edges:
#         adj_sets[u].add(v)
#         adj_sets[v].add(u)
#     used = [False] * (n+1)
#     independent_sets = []
#     for _ in range(n):
#         available = [i for i in range(1, n+1) if not used[i]]
#         if not available:
#             break
#         cur_set = []
#         for v in available:
#             if all((v not in adj_sets[u]) for u in cur_set):
#                 cur_set.append(v)
#         for v in cur_set:
#             used[v] = True
#         if cur_set:
#             independent_sets.append(cur_set)
#     return independent_sets

# def coloring_sat_upper_bound(n, edges):
#     independent_sets = greedy_coloring_sets(n, edges)
#     upper_bound = len(independent_sets)
#     print(f"Initial coloring upper bound: {upper_bound}")
#     for k in range(upper_bound, 0, -1):
#         print(f"Testing clique size k = {k} ... ", end='')
#         solver = Glucose3()
#         var = lambda i: i
#         adj_sets = [set() for _ in range(n+1)]
#         for u, v in edges:
#             adj_sets[u].add(v)
#             adj_sets[v].add(u)
#         for i in range(1, n+1):
#             for j in range(i+1, n+1):
#                 if j not in adj_sets[i]:
#                     solver.add_clause([-var(i), -var(j)])
#         xvars = [var(i) for i in range(1, n+1)]
#         cnf = CardEnc.equals(lits=xvars, bound=k, encoding=1)
#         for clause in cnf.clauses:
#             solver.add_clause(clause)
#         for indep in independent_sets:
#             if len(indep) > 1:
#                 solver.add_clause([-var(i) for i in indep])
#         if solver.solve():
#             print("SAT (tồn tại clique)")
#         else:
#             print("UNSAT ⇒ Upper bound = ", k-1)
#             return k-1
#     return 0

# def degree_upper_bound(n, edges):
#     degree = [0] * (n + 1)
#     for u, v in edges:
#         degree[u] += 1
#         degree[v] += 1
#     return max(degree)

# def smart_upper_bound(n, edges):
#     deg_ub = degree_upper_bound(n, edges)
#     col_ub = len(greedy_coloring_sets(n, edges))
#     upper = min(deg_ub + 1, col_ub)
#     print(f"Degree upper bound: {deg_ub + 1}, Coloring upper bound: {col_ub}")
#     if upper <= 100:
#         print("SAT-coloring upper bound (refine)...")
#         ub_sat = coloring_sat_upper_bound(n, edges)
#         return min(ub_sat, upper)
#     else:
#         print("Skip SAT-coloring (upper bound quá lớn)")
#         return upper

def complement_adj_matrix(n, edges):
    # Xây dựng ma trận kề của đồ thị gốc
    adj = np.zeros((n, n), dtype=int)
    for u, v in edges:
        adj[u-1, v-1] = 1
        adj[v-1, u-1] = 1
    # Ma trận kề của đồ thị bổ sung: 1 ở các vị trí không có cạnh, 0 ở đường chéo
    comp = np.ones((n, n), dtype=int) - adj - np.eye(n, dtype=int)
    return comp

def clique_upper_bound_complement_rank(n, edges):
    comp = complement_adj_matrix(n, edges)
    r = np.linalg.matrix_rank(comp)
    # Sử dụng công thức có nghĩa hơn:
    # Bound 1: n - r (nhưng thường rất lỏng)
    # Bound 2: ⌊ (1 + sqrt(1 + 8*r)) / 2 ⌋ (từ bài báo, tight hơn với rank nhỏ)
    bound1 = n - r
    bound2 = int((1 + np.sqrt(1 + 8*r)) // 2)
    print(f"Rank of complement adjacency: {r}, Bound1 (n-r): {bound1}, Bound2 (sqrt-formula): {bound2}")
    # Lấy min với n vì không thể lớn hơn số đỉnh
    return min(bound1, bound2, n)

def eigenvalue_upper_bound(n, edges):
    G = nx.Graph()
    G.add_nodes_from(range(1, n+1))
    G.add_edges_from(edges)
    adj = nx.adjacency_matrix(G).toarray()
    eigenvalues = eigvals(adj)
    n_minus_one = sum(1 for ev in eigenvalues.real if ev <= -1)
    return n_minus_one + 1

def MCP(cnf_file, n, edges, k,timeout=300 ):
    # Convert edges to a more efficient data structure
    adj_sets = [set() for _ in range(n+1)]
    for u, v in edges:
        adj_sets[u].add(v)
        adj_sets[v].add(u)  # Since the graph is undirected
    
    # Find non-adjacent pairs more efficiently
    not_edges = [(i, j) for i in range(1, n+1) for j in range(i+1, n+1) if j not in adj_sets[i]]
    
    clauses = []
    var = lambda i: i  # Biến x_i: 1..n
    # Constraint 1: Không có cạnh nối -> không cùng thuộc clique
    for i, j in not_edges:
        clauses.append([-var(i), -var(j)])
    # Constraint 2: Ràng buộc kích thước clique
    def c_var(i, j):
        return n + (i-1)*(k+1) + j + 1  # Biến phụ c_{i,j} với i in [1,n], j in [0,k]
    clauses.append([c_var(1, 0)])
    clauses.append([-c_var(1, 1), var(1)])
    clauses.append([-var(1), c_var(1, 1)])
    for j in range(2, k+1):
        clauses.append([-c_var(1, j)])
    for i in range(2, n+1):
        for j in range(0, k+1):
            clauses.append([-c_var(i-1, j), c_var(i, j)])
            if j > 0:
                clauses.append([-var(i), -c_var(i-1, j-1), c_var(i, j)])
                clauses.append([-c_var(i, j), c_var(i-1, j), var(i)])
                clauses.append([-c_var(i, j), c_var(i-1, j), c_var(i-1, j-1)])
            else:
                clauses.append([-c_var(i, j), c_var(i-1, j)])
    clauses.append([c_var(n, k)])
    solver = Glucose3()
    # solver.set_timeout(timeout)  # Set a timeout of 5 minutes
    for cl in clauses:
        solver.add_clause(cl)

        solve_start = time.time()
        
    # if solver.solve():
    #     solve_time = time.time() - solve_start
    #     print(f"SAT! Solved in {solve_time:.2f} seconds")
    #     model = solver.get_model()
    #     clique = [i for i in range(1, n+1) if var(i) in model]
    #     return clique
    # else:
    #     solve_time = time.time() - solve_start
    #     print(f"UNSAT! Proved in {solve_time:.2f} seconds")
    #     return None
    if solver.solve():
        model = solver.get_model()
        clique = [i for i in range(1, n+1) if var(i) in model]
        return clique
    else:
        return None

# if __name__ == "__main__":
#     n, edges = read_clq_file("C125.9.clq.txt")
#     max_clique = []
    
#     print(f"Graph has {n} vertices and {len(edges)} edges")
    
#     # range
#     left, right = 30, 35  
#     best_k = 0
    
#     while left <= right:
#         k = (left + right) // 2
#         print(f"\n--- Trying to find a clique of size {k} ---")
#         clique = MCP(None, n, edges, k, timeout=300)
        
#         if clique is not None:
#             max_clique = clique
#             best_k = k
#             print(f"✓ Found a clique of size {k}")
#             left = k + 1  # Try larger
#         else:
#             print(f"✗ No clique of size {k} exists")
#             right = k - 1  # Try smaller
    
#     # Print the final result
#     if max_clique:
#         print("\n=== RESULTS ===")
#         print(f"Maximum clique size: {best_k}")
#         print(f"Clique vertices: {sorted(max_clique)}")
        
        # Verify the solution is correct

if __name__ == "__main__":
    fname = "instances/keller4.clq.txt"
    n, edges = read_clq_file(fname)
    results = []

    
    print(f"\n===== FILE: {fname} | {n} vertices, {len(edges)} edges =====")

    lb = greedy_clique(n, edges)

        # n2, edges2, old2new, valid_nodes = prune_by_degree(n, edges, lb)
        # print(f"After pruning (degree >= {lb-1}): {n2} nodes, {len(edges2)} edges")

        # if n2 == 0:
        #     print("No nodes after pruning!")
        #     continue

    # n2, edges2, old2new, selected = top_degree_subgraph(n, edges, top_k=150)
    # print(f"Subgraph: {n2} nodes (top degree), {len(edges2)} edges")


        # 3. Upper bound trên graph đã prune
    ub = eigenvalue_upper_bound(n, edges)
    print(f"Lower bound: {lb} | Upper bound : {ub}")
        
    left, right = lb, ub
    max_clique = []
    best_k = 0
    L, R = left, right
    start = time.time()
        
    while L <= R:
        k = (L + R) // 2
        print(f"Trying k={k}...", end=' ')
        clique = MCP(None, n, edges, k, timeout=300)
        if clique is not None:
            print("✓", end="  ")
            max_clique = clique
            best_k = k
            L = k + 1
        else:
            print("✗", end="  ")
            R = k - 1
    total_time = time.time() - start
    if max_clique:
        print(f"\nMax clique size: {best_k} | Vertices: {sorted(max_clique)} | Time: {total_time:.2f}s")
    else:
        print("No clique found in this range!")       
    #     adj_sets = [set() for _ in range(n+1)]
    #     for u, v in edges:
    #         adj_sets[u].add(v)
    #         adj_sets[v].add(u)
            
    #     is_valid = True
    #     for i in max_clique:
    #         for j in max_clique:
    #             if i != j and j not in adj_sets[i]:
    #                 is_valid = False
    #                 print(f"ERROR: Vertices {i} and {j} are in the clique but not connected!")
    #                 break
    #         if not is_valid:
    #             break
                
    #     if is_valid:
    #         print("Verification: This is a valid clique ✓")
    # else:
    #     print("No clique found!")