import heapq


def build_reverse_edges(edges):
    reverse_edges = {}

    for start_node, neighbours in edges.items():
        for end_node, cost in neighbours:
            if end_node not in reverse_edges:
                reverse_edges[end_node] = []
            reverse_edges[end_node].append((start_node, cost))

    return reverse_edges


def bidirectional_search(edges, origin, destinations):
    """
    Custom Search 2, bidirectional uniform-cost style search.

    Forward search starts from the origin using normal directed edges.
    Backward search starts from all destination nodes using reversed directed edges.
    This allows it to work correctly for directed graphs.
    """

    reverse_edges = build_reverse_edges(edges)

    forward_frontier = [(0, origin, [origin])]
    backward_frontier = []

    for destination in destinations:
        heapq.heappush(backward_frontier, (0, destination, [destination]))

    forward_visited = {origin: (0, [origin])}
    backward_visited = {destination: (0, [destination]) for destination in destinations}

    nodes_created = 1 + len(destinations)

    while forward_frontier and backward_frontier:
        forward_cost, forward_node, forward_path = heapq.heappop(forward_frontier)

        if forward_node in backward_visited:
            backward_path = backward_visited[forward_node][1]
            full_path = forward_path + backward_path[1:]
            return full_path[-1], nodes_created, full_path

        for neighbour, weight in sorted(edges.get(forward_node, []), key=lambda x: x[0]):
            new_cost = forward_cost + weight

            if neighbour not in forward_visited or new_cost < forward_visited[neighbour][0]:
                new_path = forward_path + [neighbour]
                forward_visited[neighbour] = (new_cost, new_path)
                heapq.heappush(forward_frontier, (new_cost, neighbour, new_path))
                nodes_created += 1

        backward_cost, backward_node, backward_path = heapq.heappop(backward_frontier)

        if backward_node in forward_visited:
            forward_path = forward_visited[backward_node][1]
            full_path = forward_path + backward_path[1:]
            return full_path[-1], nodes_created, full_path

        for neighbour, weight in sorted(reverse_edges.get(backward_node, []), key=lambda x: x[0]):
            new_cost = backward_cost + weight

            if neighbour not in backward_visited or new_cost < backward_visited[neighbour][0]:
                new_path = [neighbour] + backward_path
                backward_visited[neighbour] = (new_cost, new_path)
                heapq.heappush(backward_frontier, (new_cost, neighbour, new_path))
                nodes_created += 1

    return None, nodes_created, None