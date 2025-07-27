import os
import time
import multiprocessing
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

def prune_by_degree(n, edges, lower_bound):
    degree = [0] * (n + 1)
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    # Giữ lại các đỉnh bậc đủ lớn
    valid_nodes = set(i for i in range(1, n+1) if degree[i] >= lower_bound - 1)
    # Map lại node index cho subgraph
    old2new = {old: idx+1 for idx, old in enumerate(sorted(valid_nodes))}
    new_edges = [(old2new[u], old2new[v]) for u, v in edges if u in valid_nodes and v in valid_nodes]
    return len(valid_nodes), new_edges, old2new, {v for v in valid_nodes}

def top_degree_subgraph(n, edges, top_k=100):
    degree = [0] * (n + 1)
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    sorted_nodes = sorted(range(1, n+1), key=lambda i: degree[i], reverse=True)
    selected = set(sorted_nodes[:top_k])
    # Remap node index để dễ MCP
    old2new = {old: idx+1 for idx, old in enumerate(sorted(selected))}
    new_edges = [(old2new[u], old2new[v]) for u, v in edges if u in selected and v in selected]
    return len(selected), new_edges, old2new, selected

def greedy_coloring_sets(n, edges):
    adj_sets = [set() for _ in range(n+1)]
    for u, v in edges:
        adj_sets[u].add(v)
        adj_sets[v].add(u)
    used = [False] * (n+1)
    independent_sets = []
    for _ in range(n):
        available = [i for i in range(1, n+1) if not used[i]]
        if not available:
            break
        cur_set = []
        for v in available:
            if all((v not in adj_sets[u]) for u in cur_set):
                cur_set.append(v)
        for v in cur_set:
            used[v] = True
        if cur_set:
            independent_sets.append(cur_set)
    return independent_sets

def coloring_sat_upper_bound(n, edges):
    independent_sets = greedy_coloring_sets(n, edges)
    upper_bound = len(independent_sets)
    print(f"Initial coloring upper bound: {upper_bound}")
    for k in range(upper_bound, 0, -1):
        print(f"Testing clique size k = {k} ... ", end='')
        solver = Glucose3()
        var = lambda i: i
        adj_sets = [set() for _ in range(n+1)]
        for u, v in edges:
            adj_sets[u].add(v)
            adj_sets[v].add(u)
        for i in range(1, n+1):
            for j in range(i+1, n+1):
                if j not in adj_sets[i]:
                    solver.add_clause([-var(i), -var(j)])
        xvars = [var(i) for i in range(1, n+1)]
        cnf = CardEnc.equals(lits=xvars, bound=k, encoding=1)
        for clause in cnf.clauses:
            solver.add_clause(clause)
        for indep in independent_sets:
            if len(indep) > 1:
                solver.add_clause([-var(i) for i in indep])
        if solver.solve():
            print("SAT (tồn tại clique)")
        else:
            print("UNSAT ⇒ Upper bound = ", k-1)
            return k-1
    return 0

def degree_upper_bound(n, edges):
    degree = [0] * (n + 1)
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    return max(degree)

def smart_upper_bound(n, edges):
    deg_ub = degree_upper_bound(n, edges)
    col_ub = len(greedy_coloring_sets(n, edges))
    upper = min(deg_ub + 1, col_ub)
    print(f"Degree upper bound: {deg_ub + 1}, Coloring upper bound: {col_ub}")
    if upper <= 100:
        print("SAT-coloring upper bound (refine)...")
        ub_sat = coloring_sat_upper_bound(n, edges)
        return min(ub_sat, upper)
    else:
        print("Skip SAT-coloring (upper bound quá lớn)")
        return upper

# ========== MCP with Timeout ==========

def MCP_worker(clauses, n, k, queue):
    from pysat.solvers import Glucose3
    solver = Glucose3()
    for cl in clauses:
        solver.add_clause(cl)
    result = solver.solve()
    if result:
        model = solver.get_model()
        clique = [i for i in range(1, n+1) if i in model]
        queue.put(clique)
    else:
        queue.put(None)

def MCP(n, edges, k, timeout=300):
    # Tạo clauses như code cũ
    adj_sets = [set() for _ in range(n+1)]
    for u, v in edges:
        adj_sets[u].add(v)
        adj_sets[v].add(u)
    not_edges = [(i, j) for i in range(1, n+1) for j in range(i+1, n+1) if j not in adj_sets[i]]
    clauses = []
    var = lambda i: i
    for i, j in not_edges:
        clauses.append([-var(i), -var(j)])
    def c_var(i, j):
        return n + (i-1)*(k+1) + j + 1
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

    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=MCP_worker, args=(clauses, n, k, queue))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        print(f"(timeout {timeout}s)", end=' ')
        return None  # Timeout
    if not queue.empty():
        return queue.get()
    return None

# ========== Main ==========

if __name__ == "__main__":
    fname = "instances/hamming10-4.clq.txt"
    n, edges = read_clq_file(fname)
    print(f"\n===== FILE: {fname} | {n} vertices, {len(edges)} edges =====")

    lb = greedy_clique(n, edges)
    n1, edges1, old2new1, valid_nodes = prune_by_degree(n, edges, lb)
    n2, edges2, old2new, selected = top_degree_subgraph(n, edges, top_k=150)
    print(f"Subgraph: {n2} nodes (top degree), {len(edges2)} edges")

    ub = smart_upper_bound(n2, edges2)
    print(f"Lower bound: {lb} | Upper bound (after pruning): {ub}")

    max_clique = []
    best_k = 0
    start = time.time()
    # Tuyến tính từ lb lên ub
    for k in range(lb, ub + 1):
        print(f"Trying k={k}...", end=' ')
        clique = MCP(n, edges, k, timeout=100)  # set timeout
        if clique is not None:
            print("✓", end="  ")
            max_clique = clique
            best_k = k
        else:
            print("✗ (timeout hoặc UNSAT)", end="  ")
            break
    total_time = time.time() - start
    if max_clique:
        print(f"\nMax clique size: {best_k} | Vertices: {sorted(max_clique)} | Time: {total_time:.2f}s")
    else:
        print("No clique found in this range!")

