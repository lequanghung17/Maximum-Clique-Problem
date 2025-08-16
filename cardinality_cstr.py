import os
import time
import multiprocessing
from pysat.solvers import Glucose3
from pysat.card import CardEnc

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
    valid_nodes = set(i for i in range(1, n+1) if degree[i] >= lower_bound - 1)
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
        clique, timeout = MCP_pbenc(n, edges, k, timeout=60)
        if clique is not None:
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

def MCP_pbenc_worker(n, edges, k, queue):
    from pysat.solvers import Glucose3
    from pysat.card import CardEnc

    solver = Glucose3()
    xvars = [i for i in range(1, n+1)]

    # Cardinality constraint: sum x_i == k
    card_cnf = CardEnc.equals(lits=xvars, bound=k, encoding=1)  # encoding=1: sequential counter
    for clause in card_cnf.clauses:
        solver.add_clause(clause)
    # Không chọn đồng thời hai đỉnh không nối nhau
    adj_sets = [set() for _ in range(n+1)]
    for u, v in edges:
        adj_sets[u].add(v)
        adj_sets[v].add(u)
    for i in range(1, n+1):
        for j in range(i+1, n+1):
            if j not in adj_sets[i]:
                solver.add_clause([-i, -j])

    result = solver.solve()
    if result:
        model = solver.get_model()
        clique = [i for i in xvars if i in model]
        queue.put(clique)
    else:
        queue.put(None)

def MCP_pbenc(n, edges, k, timeout=300):
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=MCP_pbenc_worker, args=(n, edges, k, queue))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        print(f"(timeout {timeout}s)", end=' ')
        return None, True
    if not queue.empty():
        return queue.get(), False
    return None, False

# ========== Main ==========

if __name__ == "__main__":
    fname = "instances/brock200_2.clq.txt"
    n, edges = read_clq_file(fname)
    print(f"\n===== FILE: {fname} | {n} vertices, {len(edges)} edges =====")

    lb = greedy_clique(n, edges)
    n2, edges2, old2new, selected = top_degree_subgraph(n, edges, top_k=150)
    print(f"Subgraph: {n2} nodes (top degree), {len(edges2)} edges")

    ub = smart_upper_bound(n2, edges2)
    print(f"Lower bound: {lb} | Upper bound (after pruning): {ub}")

    max_clique = []
    best_k = 0
    start = time.time()
    # Duyệt tuyến tính từ lb lên ub với PBEnc
    for k in range(lb, ub + 1):
        print(f"Trying k={k}...", end=' ')
        clique, timeout = MCP_pbenc(n, edges, k, timeout=60)
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
