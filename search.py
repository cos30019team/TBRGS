import sys

from algorithms.BFS import BFS
from algorithms.DFS import DFS
from algorithms.GBFS import GBFS
from algorithms.AS import astar
from algorithms.CUS1 import Dijkstra_Algorithm
from algorithms.CUS2 import bidirectional_search

def parse_file(filename):
    nodes = {}
    edges = {}
    origin = None
    destinations = []
    
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
        
        current_section = None
        for i, line in enumerate(lines):
            if line.startswith("Nodes:"):
                current_section = "nodes"
                continue
            elif line.startswith("Edges:"):
                current_section = "edges"
                continue
            elif line.startswith("Origin:"):
                origin = int(lines[i+1])
                continue
            elif line.startswith("Destinations:"):
                dest_line = lines[lines.index(line)+1]
                destinations = [int(x.strip()) for x in dest_line.split(';')]
                continue
            
            if current_section == "nodes" and ":" in line:
                # Format 1: (4,1)
                node_part, coord_part = line.split(":")
                node_id = int(node_part.strip())
                # Clean up parentheses and split
                coords = coord_part.replace("(", "").replace(")","").split(",")
                nodes[node_id] = (int(coords[0].strip()), int(coords[1].strip()))
            
            elif current_section == "edges" and ":" in line:
                pair_part, cost_part = line.split(":")
                # Clean up parentheses and split only the first two values
                pair = pair_part.replace("(", "").replace(")", "").split(",")
                # u: start node 
                # v: end node
                u, v = int(pair[0].strip()), int(pair[1].strip())
                cost = float(cost_part.strip())
                
                if u not in edges:
                    edges[u] = []
                edges[u].append((v, cost))
                
                
    return nodes, edges, origin, destinations

def main():
    if len(sys.argv) < 3:
        print("Usage: python search.py <filename> <method>")
        return
    
    filename = sys.argv[1]
    method = sys.argv[2].upper()
    
    try:
        nodes, edges, origin, destinations = parse_file(filename)
    except Exception as e:
        print(f"Error parsing file: {e}")
        return
    
    # Algorithm Selection
    goal_node, nodes_count, path = None, 0, None
    if method == "BFS":
        goal_node, nodes_count, path = BFS(edges, origin, destinations)
    elif method == "DFS":
        goal_node, nodes_count, path = DFS(edges, origin, destinations)
    elif method == "GBFS":
        goal_node, nodes_count, path = GBFS(nodes, edges, origin, destinations)
    elif method in ["AS", "ASTAR"]:
        goal_node, nodes_count, path = astar(nodes, edges, origin, destinations)
    elif method == "CUS1":
        goal_node, nodes_count, path = Dijkstra_Algorithm(edges, origin, destinations)
    elif method == "CUS2":
        goal_node, nodes_count, path = bidirectional_search(edges, origin, destinations)
    
    # Output formatting
    if goal_node:
        print(f"{filename} {method}")
        print(f"{goal_node} {nodes_count}")

        path_str = " -> ".join(map(str, path))
        print(path_str)

    else:
        print(f"{filename} {method}")
        print("No path found")
        
if __name__ == "__main__":
    main()