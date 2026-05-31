import heapq

def Dijkstra_Algorithm(edges, origin, destinations):
    
    # Uninformed method 
    # Priorities nodes with the lowest path cost g(n)
    
    """
    frontier stores: (path_cost, node_id, path_list)
    path_cost g(n)
    node_id
    """
    
    frontier = []
    visited = {}
    nodes_created = 0
    max_frontier_size = 0
    nodes_expanded = 0
    
    heapq.heappush(frontier, (0, origin, [origin]))
    #print(f"\n--- TRACE START: {origin} to {destinations} ---")
    
    while frontier:
        # Pop the node with the lowest g(n)
        max_frontier_size = max(max_frontier_size, len(frontier))
        curr_g, curr_node, path = heapq.heappop(frontier)
        nodes_expanded += 1
        
        if curr_node in destinations:
            #print(f"GOAL FOUND: {curr_node}")
            #print(f"Max Frontier Size (Space): {max_frontier_size}")
            #print(f"Total Nodes Expanded (Time): {nodes_expanded}")
            return curr_node, nodes_created, path
        
        if curr_node in visited and visited[curr_node] <= curr_g:
            continue
        visited[curr_node] = curr_g
        
        raw_neighbors = edges.get(curr_node, [])
        
        # 2. Sort by neighbor ID (x[0]) to ensure consistent tie-breaking
        # This keeps the tuples intact: [(ID, Weight), (ID, Weight)]
        sorted_neighbors = sorted(raw_neighbors, key=lambda x: x[0])
        
        for neighbor, w in sorted_neighbors:
            g_n = curr_g + w
            
            if neighbor not in visited or visited[neighbor] > g_n:
                nodes_created += 1
                heapq.heappush(frontier, (g_n, neighbor, path + [neighbor]))
        
    return None, nodes_created, None
        
        