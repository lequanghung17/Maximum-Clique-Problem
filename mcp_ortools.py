from ortools.sat.python import cp_model
import os
import pandas as pd
import time

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
    solver.parameters.max_time_in_seconds = 1000.0
    start_time = time.time()
    status = solver.Solve(model)
    solve_time = time.time() - start_time

    if status == cp_model.OPTIMAL:
        clique = [i+1 for i in range(n) if solver.Value(x[i])]
        clique_size = len(clique)
        print(f"Maximum clique size: {clique_size}")
        print(f"Clique: {clique}")
        return clique_size, solve_time, "OPTIMAL"
    elif status == cp_model.FEASIBLE:
        clique = [i+1 for i in range(n) if solver.Value(x[i])]
        clique_size = len(clique)
        print(f"Timeout! Best clique found: {clique_size}")
        print(f"Clique: {clique}")
        return clique_size, solve_time, "TIMEOUT"
    else:
        print("No clique found!")
        return 0, solve_time, "NO_SOLUTION"

def run_all_instances():
    input_folder = "input"
    results = []
    
    # Lấy tất cả file .clq.txt từ folder input
    clq_files = [f for f in os.listdir(input_folder) if f.endswith('.clq.txt')]
    
    print(f"Found {len(clq_files)} instance files:")
    for file in clq_files:
        print(f"  - {file}")
    
    print("\nStarting to solve instances...\n")
    
    for filename in clq_files:
        filepath = os.path.join(input_folder, filename)
        print(f"Solving {filename}...")
        
        try:
            # Đọc instance
            n, edges = read_clq_file(filepath)
            
            # Giải bài toán
            clique_size, solve_time, status = solve_maximum_clique_or_tools(n, edges)
            
            # Lưu kết quả
            results.append({
                'Instance': filename,
                'Nodes': n,
                'Edges': len(edges),
                'Clique_Size': clique_size,
                'Solve_Time': round(solve_time, 4),
                'Status': status
            })
            
            print(f"  Result: clique size = {clique_size}, time = {solve_time:.4f}s, status = {status}\n")
            
        except Exception as e:
            print(f"  Error solving {filename}: {e}\n")
            results.append({
                'Instance': filename,
                'Nodes': 'Error',
                'Edges': 'Error',
                'Clique_Size': 'Error',
                'Solve_Time': 'Error',
                'Status': 'Error'
            })
    
    # Xuất kết quả ra file Excel
    df = pd.DataFrame(results)
    excel_filename = "rs_ortool.xlsx"
    df.to_excel(excel_filename, index=False)
    print(f"Results saved to {excel_filename}")
    
    # In tóm tắt kết quả
    print("\n=== SUMMARY ===")
    print(f"Total instances: {len(clq_files)}")
    print(f"Successfully solved: {len([r for r in results if r['Status'] != 'Error'])}")
    print(f"Errors: {len([r for r in results if r['Status'] == 'Error'])}")
    
    return results

if __name__ == "__main__":
    run_all_instances()





#     Solving brock200_2.clq.txt...
# Maximum clique size: 12
# Clique: [27, 48, 55, 70, 105, 120, 121, 135, 145, 149, 158, 183]       
#   Result: clique size = 12, time = 7.5198s, status = OPTIMAL

# Solving brock200_4.clq.txt...
# Maximum clique size: 17
# Clique: [12, 19, 28, 29, 38, 54, 65, 71, 79, 93, 117, 127, 139, 161, 165, 186, 192]
#   Result: clique size = 17, time = 339.0224s, status = OPTIMAL

# Solving brock400_2.clq.txt...
# Timeout! Best clique found: 25
# Clique: [43, 89, 90, 105, 114, 117, 129, 131, 135, 168, 199, 205, 242, 261, 272, 275, 299, 323, 338, 346, 378, 391, 392, 398, 399]
#   Result: clique size = 25, time = 1000.5091s, status = TIMEOUT

# Solving brock400_4.clq.txt...
# Timeout! Best clique found: 33
# Clique: [7, 8, 17, 19, 112, 135, 147, 154, 157, 161, 186, 197, 202, 211, 241, 242, 245, 247, 266, 267, 270, 294, 324, 334, 340, 343, 353, 362, 380, 389, 393, 394, 396]
#   Result: clique size = 33, time = 1000.8903s, status = TIMEOUT

# Solving brock800_2.clq.txt...
# Timeout! Best clique found: 20
# Clique: [4, 33, 75, 117, 122, 180, 211, 226, 261, 366, 377, 396, 433, 499, 540, 617, 622, 651, 657, 690]
#   Result: clique size = 20, time = 1002.1884s, status = TIMEOUT

# Solving DSJC500_5.clq.txt...
# Timeout! Best clique found: 13
# Clique: [101, 151, 162, 191, 193, 200, 207, 249, 281, 350, 354, 363, 435]
#   Result: clique size = 13, time = 1020.6848s, status = TIMEOUT

# Solving hamming8-4.clq.txt...
# Maximum clique size: 16
# Clique: [1, 16, 52, 61, 86, 91, 103, 106, 151, 154, 166, 171, 196, 205, 241, 256]
#   Result: clique size = 16, time = 0.3681s, status = OPTIMAL

# Solving keller4.clq.txt...
# Maximum clique size: 11
# Clique: [8, 12, 43, 65, 70, 91, 102, 119, 155, 167, 171]
#   Result: clique size = 11, time = 1.5911s, status = OPTIMAL

# Solving p_hat300-1.clq.txt...
# Maximum clique size: 8
# Clique: [115, 122, 133, 174, 190, 200, 250, 299]
#   Result: clique size = 8, time = 18.9142s, status = OPTIMAL

# Solving p_hat300-2.clq.txt...
# Maximum clique size: 25
# Clique: [19, 20, 21, 26, 38, 40, 48, 49, 54, 56, 75, 76, 139, 149, 165, 174, 199, 205, 237, 255, 259, 273, 281, 290, 296]
#   Result: clique size = 25, time = 291.4948s, status = OPTIMAL

# Solving p_hat300-3.clq.txt...
# Timeout! Best clique found: 36
# Clique: [4, 19, 20, 21, 24, 33, 40, 49, 56, 76, 89, 98, 139, 149, 160, 166, 174, 176, 190, 199, 205, 219, 221, 226, 235, 239, 245, 247, 252, 255, 273, 290, 293, 297, 298, 299]
#   Result: clique size = 36, time = 1000.4142s, status = TIMEOUT        

# Solving p_hat700-1.clq.txt...
# Timeout! Best clique found: 11
# Clique: [117, 151, 306, 334, 397, 459, 513, 528, 537, 559, 686]        
#   Result: clique size = 11, time = 1001.6635s, status = TIMEOUT
