from ortools.sat.python import cp_model

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

def solve_maximum_clique_or_tools(n, edges):
    model = cp_model.CpModel()
    x = [model.NewBoolVar(f'x_{i+1}') for i in range(n)]

 
    edge_set = set((min(u, v), max(u, v)) for u, v in edges)

    # Ràng buộc: 
    for i in range(n):
        for j in range(i+1, n):
            if (i+1, j+1) not in edge_set:
                model.AddBoolOr([x[i].Not(), x[j].Not()])

    model.Maximize(sum(x))

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        clique = [i+1 for i in range(n) if solver.Value(x[i])]
        print("Maximum clique size:", len(clique))
        print("Clique:", clique)
        return clique
    else:
        print("No clique found!")
        return None

if __name__ == "__main__":
    n, edges = read_clq_file("C125.9.clq.txt")
    solve_maximum_clique_or_tools(n, edges)
