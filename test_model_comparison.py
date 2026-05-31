from GUI import ExactMapRepository


def run_model(model):
    repo = ExactMapRepository()
    routes = repo.top_k_routes(
        origin="2000",
        dest="3002",
        date_text="2006-10-01",
        time_text="08:00",
        model=model,
        k=5,
    )

    print()
    print("=" * 60)
    print("MODEL:", model)
    print("GRU hits:", repo.gru_hits)
    print("LSTM hits:", repo.lstm_hits)
    print("Historical fallback hits:", repo.historical_hits)

    for i, r in enumerate(routes, start=1):
        print(
            f"Route {i}: "
            f"time={r['estimated_time_min']} min, "
            f"distance={r['distance_km']} km, "
            f"avg_flow={r['avg_flow']} veh/hr, "
            f"path={' -> '.join(r['path'])}"
        )


def main():
    for model in ["Best Available", "GRU", "LSTM", "Historical Avg"]:
        run_model(model)


if __name__ == "__main__":
    main()
