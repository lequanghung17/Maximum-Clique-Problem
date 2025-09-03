#  MaxSAT (PySAT RC2)

import time
import multiprocessing
from itertools import combinations
import numpy as np
import networkx as nx
from scipy.linalg import eigvals
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2


def read_clq_file(filename):
   edges = []
   n = None
   with open(filename, 'r') as f:
      for line in f:
         if not line or line.startswith('c'):
            continue
         if line.startswith('p'):
            parts = line.strip().split()
            n = int(parts[2])
         elif line.startswith('e'):
            _, u, v = line.strip().split()
            u, v = int(u), int(v)
            if u != v:
               a, b = (u, v) if u < v else (v, u)
               edges.append((a, b))
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


# Worker: MaxSAT (RC2) cho Maximum Clique
# - Hard: non-edge => (-u v) là (-u ∨ -v)
# - Soft: (v) với weight=1

def worker_maxsat(n, edges, queue):
   # Tập tất cả cặp 1..n
   all_pairs = set(combinations(range(1, n + 1), 2))
   edge_set = set(edges)
   # đảm bảo edge_set chỉ chứa (min, max)
   edge_set = set((a, b) if a < b else (b, a) for a, b in edge_set)
   non_edges = all_pairs - edge_set

   wcnf = WCNF()

   # Hard clauses:
   for u, v in non_edges:
      wcnf.append([-u, -v], weight=None)

   # Soft clauses: 
   for v in range(1, n + 1):
      wcnf.append([v], weight=1)

   with RC2(wcnf) as rc2:
      model = rc2.compute()     
    
      chosen = set(l for l in model if l > 0 and 1 <= l <= n)
      clique = sorted(chosen)
      queue.put(clique)


def MCP_maxsat(n, edges, timeout=300):
   queue = multiprocessing.Queue()
   p = multiprocessing.Process(target=worker_maxsat, args=(n, edges, queue))
   p.start()
   p.join(timeout)
   if p.is_alive():
      p.terminate()
      p.join()
      return None, True
   if not queue.empty():
      return queue.get(), False
   return None, False

 # main
if __name__ == "__main__":
   fname = "instances/keller4.clq.txt"  
   n, edges = read_clq_file(fname)
   print(f"\n===== FILE: {fname} | {n} vertices, {len(edges)} edges =====")

   
   lb = greedy_clique(n, edges)
   ub = eigenvalue_upper_bound(n, edges)
   print(f"Lower bound (greedy): {lb} | Upper bound: {ub}")

   start = time.time()
   clique, timeout = MCP_maxsat(n, edges, timeout=600)
   elapsed = time.time() - start

   if timeout:
      print(f"\nResult: TIMEOUT")
   elif clique is None:
      print(f"\nResult: No solution (unexpected for MaxSAT)")
   else:
      print(f"\nMax clique size: {len(clique)}")
      print(f"Vertices: {clique}")
      print(f"Time: {elapsed:.2f}s")
