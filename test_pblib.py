import os
import time
import multiprocessing
from pysat.solvers import Glucose3
from pysat.pb import PBEnc

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

def MCP_pbenc_worker(n, edges, k, queue):
    from pysat.solvers import Glucose3
    from pysat.pb import PBEnc

    solver = Glucose3()
    xvars = [i for i in range(1, n+1)]
    # Cardinality constraint: sum x_i == k
    card_cnf = PBEnc.equals(lits=xvars, bound=k, encoding=0)  # encoding=0 là Sequential
    for clause in card_cnf.clauses:
        solver.add_clause(clause)
    # Không chọn hai đỉnh không nối nhau
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

# MAIN
if __name__ == "__main__":
    fname = "instances/keller4.clq.txt"
    n, edges = read_clq_file(fname)
    print(f"\n===== FILE: {fname} | {n} vertices, {len(edges)} edges =====")
    lb = greedy_clique(n, edges)
    print(f"Lower bound (greedy): {lb}")

    max_clique = []
    best_k = 0
    start = time.time()
    for k in range(lb, n+1):
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
