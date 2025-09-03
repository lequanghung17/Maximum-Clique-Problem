# is

import time
import multiprocessing
from itertools import combinations
from pysat.solvers import Glucose3
from pysat.card import CardEnc, EncType

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

def greedy_clique_lb(n, edges):
   adj = [set() for _ in range(n + 1)]
   for u, v in edges:
      adj[u].add(v)
      adj[v].add(u)
   best = 0
   for s in range(1, n + 1):
      C = [s]
      for v in range(1, n + 1):
         if v not in C and all(v in adj[u] for u in C):
            C.append(v)
      best = max(best, len(C))
   return best

def degree_ub(n, edges):
   deg = [0] * (n + 1)
   for u, v in edges:
      deg[u] += 1
      deg[v] += 1
   return max(deg) + 1 if n > 0 else 0

# ============================================================
# Base CNF: non-edges; cộng thêm "guarded" (∑ x ≥ k) bằng selector
# Mỗi ràng buộc cardinality cho một k sẽ được "gắn" selector s_k:
# Mọi mệnh đề trong encoding đều thêm literal (-s_k) => (¬s_k ∨ clause)
# Khi solve(assumptions=[s_k]) => kích hoạt ràng buộc của k.
# ============================================================
def build_base_solver(n, edges):
   S = Glucose3()
   all_pairs = set(combinations(range(1, n + 1), 2))
   edge_set = set((a, b) if a < b else (b, a) for a, b in edges)
   non_edges = all_pairs - edge_set
   for u, v in non_edges:
      S.add_clause([-u, -v])  # không thể chọn đồng thời 2 đỉnh không kề
   return S

def add_atleast_k_guarded(S, xvars, k, top_id, encoding=EncType.seqcounter):
   """
   Tạo ràng buộc ∑ x >= k với selector s_k (biến mới).
   Trả về (selector_lit, new_top_id).
   """
   # cấp 1 biến selector mới
   selector = top_id + 1
   top = selector

   # encode cardinality
   enc = CardEnc.atleast(lits=xvars, bound=k, encoding=encoding, top_id=top)
   top = enc.nv

   # thêm (¬selector ∨ clause) để "kích hoạt" bằng assumptions
   for cls in enc.clauses:
      S.add_clause(cls + [-selector])

   return selector, top


# tăng dần k và dùng assumptions để bật ràng buộc tương ứng
 
def worker_incremental(n, edges, lb, ub, queue, encoding=EncType.seqcounter):
   S = build_base_solver(n, edges)
   xvars = list(range(1, n + 1))  # map đỉnh i -> biến i
   top_id = n                     # biến mới bắt đầu sau n

   best_clique = []
   best_k = 0
   selectors = {}                 # k -> selector literal

   # tăng dần k, mỗi k thêm 1 bộ mệnh đề guarded và solve(assumptions=[selector])
   for k in range(max(1, lb), max(lb, ub) + 1):
      if k not in selectors:
         sel, top_id = add_atleast_k_guarded(S, xvars, k, top_id, encoding=encoding)
         selectors[k] = sel

      sat = S.solve(assumptions=[selectors[k]])
      if sat:
         model = set(l for l in S.get_model() if l > 0)
         clique = sorted(v for v in range(1, n + 1) if v in model)
         best_clique = clique
         best_k = k
      else:
         break

   queue.put((best_k, best_clique))


def MCP_incremental(n, edges, lb, ub, timeout=300):
   queue = multiprocessing.Queue()
   p = multiprocessing.Process(target=worker_incremental, args=(n, edges, lb, ub, queue))
   p.start()
   p.join(timeout)
   if p.is_alive():
      p.terminate()
      p.join()
      return None, None, True
   if not queue.empty():
      k, clique = queue.get()
      return k, clique, False
   return None, None, False


if __name__ == "__main__":
   fname = "instances/brock200_2.clq.txt"  
   n, edges = read_clq_file(fname)
   print(f"\n===== FILE: {fname} | {n} vertices, {len(edges)} edges =====")

   lb = greedy_clique_lb(n, edges)
   ub = degree_ub(n, edges)
   print(f"Lower bound (greedy): {lb} | Upper bound (deg+1): {ub}")

   start = time.time()
   best_k, clique, timeout = MCP_incremental(n, edges, lb, ub, timeout=600)
   elapsed = time.time() - start

   if timeout:
      print("\nResult: TIMEOUT")
   elif best_k is None:
      print("\nResult: No result")
   else:
      print(f"\nMax clique size: {best_k}")
      print(f"Vertices: {clique}")
      print(f"Time: {elapsed:.2f}s")
