from collections import deque

def BFS(edges, origin, destinations):
    
    # Keep track of visited nodes
    visited = set()
    
    # Initialize frontier
    # Frontier stores: (current_node, path_to_node)
    frontier = deque([(origin, [origin])]) # Deque the next path
    
    nodes_created = 1
    
    visited.add(origin)
    
    while frontier:
        curr_node, path = frontier.popleft()
        
        # Check if we reached any of the goals
        if curr_node in destinations:
            return curr_node, nodes_created, path
        
        # nodes are expanded in ascending order of their node IDs
        neighbors = sorted([n for (n, _) in edges.get(curr_node, [])])
        
        for neighbor_id in neighbors:
            if neighbor_id not in visited:
                # Create the new path:
                nodes_created += 1
                
                visited.add(neighbor_id)
                frontier.append((neighbor_id, path + [neighbor_id]))
                
    # Frontier is empty
    return None, nodes_created, None # node goal, nodes_created, path_origin_goal
        