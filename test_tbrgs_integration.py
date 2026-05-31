from GUI import ExactMapRepository


def main():
    repo = ExactMapRepository()

    origin = "2000"
    destination = "3002"
    date = "2006-10-01"
    time = "08:00"
    model = "Best Available"
    top_k = 5

    routes = repo.top_k_routes(origin, destination, date, time, model, top_k)

    print("TBRGS Integration Test")
    print("----------------------")
    print(f"Origin: {origin}")
    print(f"Destination: {destination}")
    print(f"Date/Time: {date} {time}")
    print(f"Model: {model}")
    print(f"Routes found: {len(routes)}")
    print(f"GRU lookups: {repo.gru_hits}")
    print(f"LSTM lookups: {repo.lstm_hits}")
    print(f"Historical fallback lookups: {repo.historical_hits}")
    print()

    if not routes:
        print("FAILED: No routes found.")
        return

    for i, route in enumerate(routes, start=1):
        path = " -> ".join(route["path"])
        print(f"Route {i}")
        print(f"  Path: {path}")
        print(f"  Estimated time: {route['estimated_time_min']} min")
        print(f"  Distance: {route['distance_km']} km")
        print(f"  Average flow: {route['avg_flow']} veh/hr")
        print()

    print("PASSED: Top-k route integration is working.")


if __name__ == "__main__":
    main()
