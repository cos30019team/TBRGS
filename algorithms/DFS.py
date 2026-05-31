def DFS(edges, origin, destinations):
    stack = [(origin, [origin])]
    
    visited = set()
    nodes_created = 1
    
    while stack:
        curr_node, path = stack.pop()
        
        if curr_node in destinations:
            return curr_node, nodes_created, path
        
        if curr_node not in visited:
            visited.add(curr_node)
            
            # nodes are expanded in descending order of their node IDs
            neighbors = sorted([n for (n, _) in edges.get(curr_node, [])], reverse=True)
        
        for neighbor_id in neighbors:
            if neighbor_id not in visited:
                nodes_created += 1
                
                # Push to stack
                stack.append((neighbor_id, path + [neighbor_id]))
                
    return None, nodes_created, None