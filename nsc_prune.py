import os
import time
import multiprocessing
from pysat.solvers import Glucose3

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
    return upper_bound

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
    return upper

# ========== New Sequential Counter (NSC) Exactly k ==========

def nsc_exactly_k(solver, xvars, k, var_counter):
    n = len(xvars)
    R = [[0] * (k + 1) for _ in range(n + 1)]
    # Gán biến phụ cho R[i][j]
    for i in range(1, n + 1):
        for j in range(1, min(i, k) + 1):
            var_counter[0] += 1
            R[i][j] = var_counter[0]

    # (1): Nếu X_i = TRUE thì R[i][1] = TRUE
    for i in range(1, n + 1):
        solver.add_clause([-xvars[i-1], R[i][1]])

    # (2): Nếu R[i-1][j] = TRUE thì R[i][j] = TRUE
    for i in range(2, n + 1):
        for j in range(1, min(i-1, k) + 1):
            solver.add_clause([-R[i-1][j], R[i][j]])

    # (3): Nếu X_i = TRUE và R[i-1][j-1] = TRUE thì R[i][j] = TRUE
    for i in range(2, n + 1):
        for j in range(2, min(i, k) + 1):
            solver.add_clause([-xvars[i-1], -R[i-1][j-1], R[i][j]])

    # (4): Nếu X_i = FALSE và R[i-1][j] = FALSE thì R[i][j] = FALSE
    for i in range(2, n + 1):
        for j in range(1, min(i-1, k) + 1):
            solver.add_clause([xvars[i-1], R[i-1][j], -R[i][j]])

    # (5): Nếu X_i = FALSE thì R[i][i] = FALSE (i <= k)
    for i in range(1, min(n, k) + 1):
        solver.add_clause([xvars[i-1], -R[i][i]])

    # (6): Nếu R[i-1][j-1] = FALSE thì R[i][j] = FALSE
    for i in range(2, n + 1):
        for j in range(2, min(i, k) + 1):
            solver.add_clause([R[i-1][j-1], -R[i][j]])

    # (7): At least k
    # solver.add_clause([R[n][k], xvars[n-1]])
    # solver.add_clause([R[n][k], R[n-1][k-1]])

    solver.add_clause([R[n-1][k], xvars[n-1]])
    solver.add_clause([R[n-1][k], R[n-1][k-1]])

    # (8): At most k
    for i in range(k+1, n+1):
        solver.add_clause([-xvars[i-1], -R[i-1][k]])

# ========== MCP with NSC and Timeout ==========

def worker_nsc(n, edges, k, queue):
    from pysat.solvers import Glucose3
    solver = Glucose3()
    xvars = [i+1 for i in range(n)]  # biến x1...xn
    var_counter = [n]  # biến phụ bắt đầu từ n+1
    # Constraint 1: Không chọn đồng thời 2 đỉnh không nối nhau
    adj_sets = [set() for _ in range(n+1)]
    for u, v in edges:
        adj_sets[u].add(v)
        adj_sets[v].add(u)
    for i in range(1, n+1):
        for j in range(i+1, n+1):
            if j not in adj_sets[i]:
                solver.add_clause([-xvars[i-1], -xvars[j-1]])
    # Constraint 2: Exactly k bằng NSC mới
    nsc_exactly_k(solver, xvars, k, var_counter)
    result = solver.solve()
    if result:
        model = solver.get_model()
        clique = [i+1 for i in range(n) if xvars[i] in model]
        queue.put(clique)
    else:
        queue.put(None)

def MCP_nsc(n, edges, k, timeout=100):
    import multiprocessing
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=worker_nsc, args=(n, edges, k, queue))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        print(f"(timeout {timeout}s)", end=' ')
        return None
    if not queue.empty():
        return queue.get()
    return None

# ========== Main ==========

if __name__ == "__main__":
    fname = "instances/brock800_4.clq.txt"   # Sửa tên file ở đây nếu cần
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
    # Tuyến tính từ lb lên ub với NSC
    for k in range(lb, ub + 1):
        print(f"Trying k={k}...", end=' ')
        clique = MCP_nsc(n, edges, k, timeout=100)  # set timeout
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
