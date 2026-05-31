import math

def flow_to_speed(flow_hourly):

    if(flow_hourly <= 351):
        return 60
    else:
        a = 1.4648375
        b = -93.75
        c = flow_hourly

        d = b*b - 4*a*c

        if(d < 0):
            d = 0

        speed = (-b + math.sqrt(d)) / (2 * a)

        speed = max(speed, 10)

        return speed
    

def travel_time_seconds(distance_m, flow_mins):

    flow_hr = flow_mins * 4
    speed_kmh = flow_to_speed(flow_hr)
    speed_ms = speed_kmh / 3.6

    return distance_m / speed_ms


print("Flow 50/15min:", travel_time_seconds(100, 50))     # expect ~6 sec
print("Flow 200/15min:", travel_time_seconds(100, 200))   # expect > 6 sec
print("Flow 400/15min:", travel_time_seconds(100, 400))   # expect >> 6 sec
