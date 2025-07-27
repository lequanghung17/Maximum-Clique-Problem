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

def MCP(cnf_file, n, edges, k):
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
    for cl in clauses:
        solver.add_clause(cl)
    if solver.solve():
        model = solver.get_model()
        clique = [i for i in range(1, n+1) if i in model]
        return clique
    else:
        return None

if __name__ == "__main__":
    n, edges = read_clq_file("C125.9.clq.txt")
    max_clique = []
    
    print(f"Graph has {n} vertices and {len(edges)} edges")
    
    # Start with a smaller k for faster testing
    for k in range(35, 32, -1):
        print(f"Trying to find a clique of size {k}...")
        clique = MCP(None, n, edges, k)
        if clique is not None:
            max_clique = clique
            print(f"Maximum clique size: {len(max_clique)}")
            print("Clique:", max_clique)
            break
    else:
        print("No clique found!")