import os
import time
import multiprocessing
from pysat.solvers import Glucose3
import numpy as np
import networkx as nx
from scipy.linalg import eigvals

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
    fname = "instances/p_hat300-3.clq.txt"   # Sửa tên file ở đây nếu cần
    n, edges = read_clq_file(fname)
    print(f"\n===== FILE: {fname} | {n} vertices, {len(edges)} edges =====")

    lb = greedy_clique(n, edges)
    ub = eigenvalue_upper_bound(n, edges)
    print(f"Lower bound: {lb} | Upper bound: {ub}")

    max_clique = []
    best_k = 0
    start = time.time()
    # Tuyến tính từ lb lên ub với NSC
    for k in range(lb, ub + 1):
        print(f"Trying k={k}...", end=' ')
        clique = MCP_nsc(n, edges, k, timeout=500)  # set timeout
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
