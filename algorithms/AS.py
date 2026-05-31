import heapq
import math

def get_euclidean_dist(node_coords, goal_coords):
    return math.sqrt((node_coords[0] - goal_coords[0])**2 + (node_coords[1] - goal_coords[1])**2)

def astar(nodes, edges, origin, destinations):
    frontier = []
    visited = {} # Stores {node_id: best_g_cost}
    nodes_created = 1
    
    # Calculate h for origin
    h_start = min(get_euclidean_dist(nodes[origin], nodes[d]) for d in destinations)
    
    # Tuple: (f_n, g_n, node_id, path)
    heapq.heappush(frontier, (h_start, 0, origin, [origin]))
    
    while frontier:
        f_curr, g_curr, curr_node, path = heapq.heappop(frontier)
        
        if curr_node in destinations:
            return curr_node, nodes_created, path
        
        # Standard A* optimization: skip if we found a better path already
        if curr_node in visited and visited[curr_node] <= g_curr:
            continue
        visited[curr_node] = g_curr
        
        # Get neighbors and sort by ID for tie-breaking
        raw_neighbors = edges.get(curr_node, [])
        sorted_neighbors = sorted(raw_neighbors, key=lambda x: x[0])
        
        for neighbor, cost in sorted_neighbors:
            g_n = g_curr + cost
            h_n = min(get_euclidean_dist(nodes[neighbor], nodes[d]) for d in destinations)
            f_n = g_n + h_n
            
            # If we haven't visited or found a cheaper way
            if neighbor not in visited or g_n < visited[neighbor]:
                nodes_created += 1
                heapq.heappush(frontier, (f_n, g_n, neighbor, path + [neighbor]))
                
    return None, nodes_created, None