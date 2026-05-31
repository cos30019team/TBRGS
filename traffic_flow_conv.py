"""
Traffic flow to travel time conversion helpers for COS30019 A2B TBRGS.
"""

import math


def flow_to_speed(flow_hourly: float) -> float:
    flow_hourly = max(0.0, float(flow_hourly))

    if flow_hourly <= 351:
        return 60.0

    a = 1.4648375
    b = -93.75
    c = flow_hourly

    d = b * b - 4 * a * c

    if d < 0:
        d = 0

    speed = (-b + math.sqrt(d)) / (2 * a)
    return max(speed, 10.0)


def travel_time_seconds(distance_m: float, flow_mins: float) -> float:
    distance_m = max(0.0, float(distance_m))
    flow_mins = max(0.0, float(flow_mins))

    flow_hr = flow_mins * 4.0
    speed_kmh = flow_to_speed(flow_hr)
    speed_ms = speed_kmh / 3.6

    if speed_ms <= 0:
        return float("inf")

    return distance_m / speed_ms


if __name__ == "__main__":
    print("Flow 50/15min:", travel_time_seconds(100, 50))
    print("Flow 200/15min:", travel_time_seconds(100, 200))
    print("Flow 400/15min:", travel_time_seconds(100, 400))
