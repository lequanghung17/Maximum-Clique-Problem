import time
from pysat.solvers import Glucose3
from pysat.formula import CNF

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

def greedy_coloring(n, edges):
    # Tô màu greedy cho chromatic number (cận trên)
    adj_sets = [set() for _ in range(n+1)]
    for u, v in edges:
        adj_sets[u].add(v)
        adj_sets[v].add(u)
    color = [0] * (n+1)
    for u in range(1, n+1):
        used = set()
        for v in adj_sets[u]:
            if color[v]:
                used.add(color[v])
        c = 1
        while c in used:
            c += 1
        color[u] = c
    return max(color)


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

if __name__ == "__main__":
    n, edges = read_clq_file("instances/keller4.clq.txt")

    lb = greedy_clique(n, edges)
    ub = greedy_coloring(n, edges)
    
    left, right = lb, ub

    max_clique = []
    
    print(f"Graph has {n} vertices and {len(edges)} edges")
    print(f"Lower bound (greedy clique): {lb} | Upper bound (greedy coloring): {ub}")
    
    # range
    best_k = 0
    
    while left <= right:
        k = (left + right) // 2
        print(f"\n--- Trying to find a clique of size {k} ---")
        clique = MCP(None, n, edges, k, timeout=300)
        
        if clique is not None:
            max_clique = clique
            best_k = k
            print(f"✓ Found a clique of size {k}")
            left = k + 1  # Try larger
        else:
            print(f"✗ No clique of size {k} exists")
            right = k - 1  # Try smaller
    
    # Print the final result
    if max_clique:
        print("\n=== RESULTS ===")
        print(f"Maximum clique size: {best_k}")
        print(f"Clique vertices: {sorted(max_clique)}")
        
        # Verify the solution is correct
        adj_sets = [set() for _ in range(n+1)]
        for u, v in edges:
            adj_sets[u].add(v)
            adj_sets[v].add(u)
            
        is_valid = True
        for i in max_clique:
            for j in max_clique:
                if i != j and j not in adj_sets[i]:
                    is_valid = False
                    print(f"ERROR: Vertices {i} and {j} are in the clique but not connected!")
                    break
            if not is_valid:
                break
                
        if is_valid:
            print("Verification: This is a valid clique ✓")
    else:
        print("No clique found!")