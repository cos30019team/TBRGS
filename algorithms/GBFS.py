import heapq
import math

# Calculate Heuristics
def get_euclidean_dist(node_coords, goal_coords):
    return math.sqrt(
        (node_coords[0] - goal_coords[0])**2 + 
        (node_coords[1] - goal_coords[1])**2 
    )

def GBFS(nodes, edges, origin, destinations):
    frontier = []
    visited = set()
    nodes_created = 1
    
    # Origin relative to the CLOSEST destination
    # h(start): distance to closest goal
    h_start = min(
        get_euclidean_dist(nodes[origin], nodes[d])
        for d in destinations
    )
        
    heapq.heappush(frontier, (h_start, origin, [origin]))
    
    while frontier:
        _, curr_node, path = heapq.heappop(frontier)
        
        if curr_node in destinations:
            return curr_node, nodes_created, path
        
        if curr_node not in visited:
            visited.add(curr_node)
        
            # nodes are expanded in ascending order of their node IDs
            neighbors = sorted([n for (n, _) in edges.get(curr_node, [])])
            # Add neighbors by h(n)
            for neighbor in neighbors:
                if neighbor not in visited:
                    # Calculate h(n) to the nearest distance
                    distances = []
                    for d in destinations:
                        distances.append(get_euclidean_dist(nodes[neighbor],nodes[d]))
                    h_n = min(distances)
                
                    nodes_created += 1
                    heapq.heappush(frontier, (h_n, neighbor, path + [neighbor])) 
            
    return None, nodes_created, None