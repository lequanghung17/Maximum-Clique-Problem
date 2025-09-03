
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

def degree_upper_bound(n, edges):
    degree = [0] * (n + 1)
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    return max(degree) + 1


def build_adj_sets(n, edges):
    adj_sets = [set() for _ in range(n+1)]
    for u, v in edges:
        adj_sets[u].add(v)
        adj_sets[v].add(u)
    return adj_sets

def MCP_sec4_worker(n, edges, k, queue):
    adj_sets = build_adj_sets(n, edges)
    degree = [len(adj_sets[i]) for i in range(n+1)]

    solver = Glucose3()
    top_id = n

    # Ánh xạ: biến SAT cho mỗi đỉnh u là id = u
    xvars = list(range(1, n+1))

    # (0) Lọc đỉnh: deg(u) < k-1 => không thể nằm trong clique size k
    for u in range(1, n+1):
        if degree[u] < k-1:
            solver.add_clause([-xvars[u-1]])

    # (1) Exactly k: ∑ x_u = k
    enc_eq = CardEnc.equals(lits=xvars, bound=k, encoding=1, top_id=top_id)
    top_id = enc_eq.nv
    for cls in enc_eq.clauses:
        solver.add_clause(cls)

    # (2) Local constraints: x_u -> ∑_{v∈Adj(u)} x_v ≥ k-1
    for u in range(1, n+1):
        if degree[u] >= k-1:  # chỉ encode cho đỉnh có deg đủ
            neigh_vars = [v for v in adj_sets[u]]
            if len(neigh_vars) < k-1:
                solver.add_clause([-u])
                continue
            enc_atl = CardEnc.atleast(lits=neigh_vars, bound=k-1,
                                      encoding=1, top_id=top_id)
            top_id = enc_atl.nv
            for cls in enc_atl.clauses:
                solver.add_clause(cls + [-u])  # (¬x_u ∨ clause)

    sat = solver.solve()
    if not sat:
        queue.put(None)
        return

    model = solver.get_model()
    clique_nodes = [u for u in range(1, n+1) if model[u-1] > 0]
    queue.put(sorted(clique_nodes))

def MCP_sec4(n, edges, k, timeout=300):
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=MCP_sec4_worker, args=(n, edges, k, queue))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return None, True
    if not queue.empty():
        return queue.get(), False
    return None, False


if __name__ == "__main__":
    fname = "instances/brock200_2.clq.txt"   
    n, edges = read_clq_file(fname)
    print(f"\n===== FILE: {fname} | {n} vertices, {len(edges)} edges =====")

    lb = greedy_clique(n, edges)
    ub = degree_upper_bound(n, edges)
    print(f"Lower bound: {lb} | Upper bound: {ub}")

    max_clique = []
    best_k = 0
    start = time.time()

    for k in range(lb, ub + 1):
        print(f"Trying k={k}...", end=" ")
        clique, timeout = MCP_sec4(n, edges, k, timeout=300)
        if clique is not None:
            print("✓")
            max_clique = clique
            best_k = k
        else:
            print("✗")
            break

    total_time = time.time() - start
    if max_clique:
        print(f"\nMax clique size: {best_k}")
        print(f"Vertices: {sorted(max_clique)}")
        print(f"Time: {total_time:.2f}s")
    else:
        print("No clique found in this range!")





























# import os
# import time
# import multiprocessing
# from pysat.solvers import Glucose3
# from pysat.card import CardEnc

# # ============================================================
# # Đọc file .clq.txt
# # ============================================================
# def read_clq_file(filename):
#     edges = []
#     n = None
#     with open(filename, 'r') as f:
#         for line in f:
#             if line.startswith('p'):
#                 parts = line.strip().split()
#                 n = int(parts[2])
#             elif line.startswith('e'):
#                 _, u, v = line.strip().split()
#                 edges.append((int(u), int(v)))
#     return n, edges

# # ============================================================
# # Các hàm cơ sở: greedy lower bound, upper bound
# # ============================================================
# def greedy_clique(n, edges):
#     adj_sets = [set() for _ in range(n+1)]
#     for u, v in edges:
#         adj_sets[u].add(v)
#         adj_sets[v].add(u)
#     best = []
#     for start in range(1, n+1):
#         clique = [start]
#         for v in range(1, n+1):
#             if v not in clique and all(v in adj_sets[u] for u in clique):
#                 clique.append(v)
#         if len(clique) > len(best):
#             best = clique
#     return len(best)

# def degree_upper_bound(n, edges):
#     degree = [0] * (n + 1)
#     for u, v in edges:
#         degree[u] += 1
#         degree[v] += 1
#     return max(degree) + 1

# # ============================================================
# # Tiện ích adjacency + lọc bậc hiệu dụng
# # ============================================================
# def build_adj_sets(n, edges):
#     adj_sets = [set() for _ in range(n + 1)]
#     for u, v in edges:
#         adj_sets[u].add(v)
#         adj_sets[v].add(u)
#     return adj_sets

# def effective_degree_prune(active_set, adj_sets, K):
#     active = set(active_set)
#     changed = True
#     while changed:
#         changed = False
#         remove = []
#         for u in list(active):
#             deg_eff = len(adj_sets[u].intersection(active))
#             if deg_eff < K - 1:
#                 remove.append(u)
#         if remove:
#             for u in remove:
#                 active.remove(u)
#             changed = True
#     return active

# # ============================================================
# # Worker: Cardinality Encoding mới
# # ============================================================
# def MCP_pbenc_worker(n, edges, k, queue):
#     """
#     Cardinality encoding mới:
#       (1) sum x = k
#       (2) với mỗi u: x_u -> sum_{v in N(u)} x_v >= k-1
#     Có lọc bậc hiệu dụng trước khi mã hoá.
#     """
#     adj_sets = build_adj_sets(n, edges)
#     active = effective_degree_prune(set(range(1, n + 1)), adj_sets, k)

#     if len(active) < k:
#         queue.put(None)
#         return

#     # ánh xạ biến SAT
#     active_sorted = sorted(active)
#     var_of = {u: i + 1 for i, u in enumerate(active_sorted)}
#     node_of = {i + 1: u for i, u in enumerate(active_sorted)}

#     solver = Glucose3()
#     top = len(active_sorted)

#     # (1) exactly k
#     xvars = [var_of[u] for u in active_sorted]
#     enc_eq = CardEnc.equals(lits=xvars, bound=k, encoding=1, top_id=top)
#     top = enc_eq.nv
#     for cls in enc_eq.clauses:
#         solver.add_clause(cls)

#     # (2) điều kiện neighborhood
#     for u in active_sorted:
#         xu = var_of[u]
#         neigh_vars = [var_of[v] for v in adj_sets[u] if v in var_of]
#         if len(neigh_vars) < k - 1:
#             solver.add_clause([-xu])
#             continue
#         enc_atl = CardEnc.atleast(lits=neigh_vars, bound=k-1, encoding=1, top_id=top)
#         top = enc_atl.nv
#         for cls in enc_atl.clauses:
#             solver.add_clause(cls + [-xu])

#     sat = solver.solve()
#     if not sat:
#         queue.put(None)
#         return

#     model = solver.get_model()
#     clique_nodes = [node_of[lit] for lit in model if lit > 0 and lit in node_of]

#     if len(clique_nodes) > k:  # phòng thừa
#         clique_nodes_sorted = []
#         for v in sorted(clique_nodes):
#             if len(clique_nodes_sorted) < k and all((v in adj_sets[u]) for u in clique_nodes_sorted):
#                 clique_nodes_sorted.append(v)
#         clique_nodes = clique_nodes_sorted

#     queue.put(sorted(clique_nodes))

# # ============================================================
# # Wrapper có timeout
# # ============================================================
# def MCP_pbenc(n, edges, k, timeout=300):
#     queue = multiprocessing.Queue()
#     p = multiprocessing.Process(target=MCP_pbenc_worker, args=(n, edges, k, queue))
#     p.start()
#     p.join(timeout)
#     if p.is_alive():
#         p.terminate()
#         p.join()
#         return None, True
#     if not queue.empty():
#         return queue.get(), False
#     return None, False

# # ============================================================
# # MAIN
# # ============================================================
# if __name__ == "__main__":
#     fname = "instances/brock200_2.clq.txt"   
#     n, edges = read_clq_file(fname)
#     print(f"\n===== FILE: {fname} | {n} vertices, {len(edges)} edges =====")

#     lb = greedy_clique(n, edges)
#     ub = degree_upper_bound(n, edges)
#     print(f"Lower bound: {lb} | Upper bound: {ub}")

#     max_clique = []
#     best_k = 0
#     start = time.time()

#     for k in range(lb, ub + 1):
#         print(f"Trying k={k}...", end=" ")
#         clique, timeout = MCP_pbenc(n, edges, k, timeout=60)
#         if clique is not None:
#             print("✓")
#             max_clique = clique
#             best_k = k
#         else:
#             print("✗")
#             break

#     total_time = time.time() - start
#     if max_clique:
#         print(f"\nMax clique size: {best_k}")
#         print(f"Vertices: {sorted(max_clique)}")
#         print(f"Time: {total_time:.2f}s")
#     else:
#         print("No clique found in this range!")
