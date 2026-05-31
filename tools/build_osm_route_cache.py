"""
Build OSM road-following geometry for the TBRGS GUI.

This script uses:
- SCATS site coordinates from data/raw/Scats Data October 2006.xls
- assignment route graph from GUI.ExactMapRepository
- OpenStreetMap road geometry via OSMnx

Output:
data/map_osm/osm_route_cache.json

Run from project root:
python tools/build_osm_route_cache.py

Arif's command:
& "C:\\Users\\AriF0019\\AppData\\Local\\Programs\\Python\\Python312\\python.exe" tools/build_osm_route_cache.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import networkx as nx
import osmnx as ox
import pandas as pd

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from GUI import ExactMapRepository  # noqa: E402


RAW_SCATS_XLS = PROJECT_ROOT / "data" / "raw" / "Scats Data October 2006.xls"
OUTPUT_DIR = PROJECT_ROOT / "data" / "map_osm"
OUTPUT_JSON = OUTPUT_DIR / "osm_route_cache.json"
OUTPUT_GRAPHML = OUTPUT_DIR / "boroondara_drive_graph.graphml"


def clean_scats(value) -> str:
    try:
        return str(int(float(value)))
    except Exception:
        return str(value).strip()


def load_scats_centroids() -> dict:
    """
    Reads the assignment SCATS Excel file and calculates one representative
    latitude/longitude coordinate per SCATS site.

    Some SCATS sites have several directional detector rows, so we average the
    valid coordinates. Invalid 0,0 rows are ignored.
    """
    if not RAW_SCATS_XLS.exists():
        raise FileNotFoundError(f"Missing file: {RAW_SCATS_XLS}")

    df = pd.read_excel(RAW_SCATS_XLS, sheet_name="Data", header=1)

    required = ["SCATS Number", "NB_LATITUDE", "NB_LONGITUDE"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns in Excel: {missing}")

    df = df[required].copy()
    df["site"] = df["SCATS Number"].apply(clean_scats)
    df["lat"] = pd.to_numeric(df["NB_LATITUDE"], errors="coerce")
    df["lon"] = pd.to_numeric(df["NB_LONGITUDE"], errors="coerce")

    # Remove invalid and zero coordinates.
    df = df.dropna(subset=["lat", "lon"])
    df = df[(df["lat"] != 0) & (df["lon"] != 0)]

    centroids = {}

    for site, group in df.groupby("site"):
        centroids[site] = {
            "lat": float(group["lat"].mean()),
            "lon": float(group["lon"].mean()),
            "detector_count": int(len(group)),
        }

    return centroids


def get_or_download_osm_graph(scats_centroids: dict):
    """
    Downloads or loads a drive road network covering the SCATS area.
    """
    if OUTPUT_GRAPHML.exists():
        print("Loading cached OSM graph:", OUTPUT_GRAPHML)
        return ox.load_graphml(OUTPUT_GRAPHML)

    lats = [v["lat"] for v in scats_centroids.values()]
    lons = [v["lon"] for v in scats_centroids.values()]

    north = max(lats) + 0.015
    south = min(lats) - 0.015
    east = max(lons) + 0.015
    west = min(lons) - 0.015

    print("Downloading OSM road network...")
    print(f"Bounding box: north={north}, south={south}, east={east}, west={west}")

    # OSMnx v2 uses bbox order: (west, south, east, north)
    graph = ox.graph_from_bbox(
        (west, south, east, north),
        network_type="drive",
        simplify=True,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(graph, OUTPUT_GRAPHML)

    print("Saved OSM graph:", OUTPUT_GRAPHML)
    return graph


def edge_geometry_points(graph, u, v, k=0) -> list:
    """
    Returns an OSM edge geometry as [[lat, lon], ...].
    If geometry is missing, uses the endpoint coordinates.
    """
    data = graph.get_edge_data(u, v)

    if data is None:
        return []

    # MultiDiGraph can have multiple keyed edges.
    if isinstance(data, dict):
        if k in data:
            edge_data = data[k]
        else:
            first_key = next(iter(data))
            edge_data = data[first_key]
    else:
        edge_data = data

    if "geometry" in edge_data and edge_data["geometry"] is not None:
        coords = list(edge_data["geometry"].coords)
        return [[float(lat), float(lon)] for lon, lat in coords]

    return [
        [float(graph.nodes[u]["y"]), float(graph.nodes[u]["x"])],
        [float(graph.nodes[v]["y"]), float(graph.nodes[v]["x"])],
    ]


def build_path_geometry(graph, route_nodes: list) -> list:
    """
    Converts an OSM node route into a continuous list of [lat, lon] points.
    """
    coords = []

    for u, v in zip(route_nodes[:-1], route_nodes[1:]):
        segment = edge_geometry_points(graph, u, v)

        if not segment:
            continue

        if coords and segment:
            # Avoid duplicating the join point.
            coords.extend(segment[1:])
        else:
            coords.extend(segment)

    return coords



def build_background_road_segments(graph, scats_centroids: dict, max_segments: int = 18000) -> list:
    """
    Extracts OSM road geometry for the offline Pygame background.

    Final map strategy:
    - keep a broad Boroondara road background so the map does not have empty holes
    - prioritise named/main roads first
    - still cap the number of road segments for performance
    - selected route geometry is stored separately and always drawn fully
    """
    scats_points = [
        (float(v["lat"]), float(v["lon"]))
        for v in scats_centroids.values()
        if float(v["lat"]) != 0 and float(v["lon"]) != 0
    ]

    def segment_points(u, v, data):
        if "geometry" in data and data["geometry"] is not None:
            return [[float(lat), float(lon)] for lon, lat in data["geometry"].coords]

        return [
            [float(graph.nodes[u]["y"]), float(graph.nodes[u]["x"])],
            [float(graph.nodes[v]["y"]), float(graph.nodes[v]["x"])],
        ]

    def approx_score(coords, data):
        """
        Lower score means more useful for the visual road background.
        """
        mid = coords[len(coords) // 2]
        lat, lon = float(mid[0]), float(mid[1])

        # Distance to nearest assignment SCATS site.
        best = 999999.0
        for s_lat, s_lon in scats_points:
            d = (lat - s_lat) ** 2 + (lon - s_lon) ** 2
            if d < best:
                best = d

        score = best

        # Prefer named roads and main roads, but do not remove local roads.
        if data.get("name"):
            score *= 0.70

        highway = str(data.get("highway", "")).lower()

        if any(h in highway for h in ["motorway", "trunk", "primary", "secondary", "tertiary"]):
            score *= 0.50
        elif "residential" in highway:
            score *= 0.85
        elif "service" in highway:
            score *= 1.20

        return score

    candidates = []

    for u, v, data in graph.edges(data=True):
        coords = segment_points(u, v, data)

        if len(coords) < 2:
            continue

        score = approx_score(coords, data)
        candidates.append((score, coords))

    candidates.sort(key=lambda item: item[0])

    return [coords for _, coords in candidates[:max_segments]]


def path_distance_km(coords: list) -> float:
    """
    Approximate path length using haversine distance.
    """
    def haversine(a, b):
        lat1, lon1 = math.radians(a[0]), math.radians(a[1])
        lat2, lon2 = math.radians(b[0]), math.radians(b[1])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        h = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        )

        return 6371.0 * 2 * math.atan2(math.sqrt(h), math.sqrt(1 - h))

    if len(coords) < 2:
        return 0.0

    return sum(haversine(a, b) for a, b in zip(coords[:-1], coords[1:]))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading assignment SCATS coordinates...")
    scats_centroids = load_scats_centroids()
    print("SCATS sites with valid coordinates:", len(scats_centroids))

    print("Loading assignment route graph from GUI repository...")
    repo = ExactMapRepository()
    graph_links = repo.graph
    print("Route graph nodes:", len(graph_links))

    # Only keep route graph sites that also have real lat/lon.
    valid_sites = sorted(set(graph_links.keys()) & set(scats_centroids.keys()))
    print("Route graph sites with coordinates:", len(valid_sites))

    osm_graph = get_or_download_osm_graph(scats_centroids)

    print("Snapping SCATS sites to nearest OSM road nodes...")
    scats_to_osm = {}

    for site in valid_sites:
        lat = scats_centroids[site]["lat"]
        lon = scats_centroids[site]["lon"]

        nearest = ox.distance.nearest_nodes(osm_graph, X=lon, Y=lat)
        scats_to_osm[site] = int(nearest)

    print("Building road-following geometry for SCATS graph edges...")

    route_paths = {}
    failed = []

    for src in valid_sites:
        for dst, assignment_distance in graph_links.get(src, []):
            dst = clean_scats(dst)

            if dst not in scats_to_osm:
                continue

            key = f"{src}->{dst}"

            try:
                start_osm = scats_to_osm[src]
                end_osm = scats_to_osm[dst]

                osm_route = nx.shortest_path(
                    osm_graph,
                    source=start_osm,
                    target=end_osm,
                    weight="length",
                )

                coords = build_path_geometry(osm_graph, osm_route)

                if not coords:
                    failed.append(key)
                    continue

                route_paths[key] = {
                    "coords": coords,
                    "osm_node_count": len(osm_route),
                    "assignment_distance_km": float(assignment_distance),
                    "osm_distance_km": round(path_distance_km(coords), 4),
                }

            except Exception as exc:
                failed.append(key)
                print("Failed:", key, exc)

    road_segments = build_background_road_segments(osm_graph, scats_centroids)

    output = {
        "source": "OpenStreetMap drive network via OSMnx, assignment SCATS coordinates from Scats Data October 2006.xls",
        "attribution": "Road network data © OpenStreetMap contributors",
        "scats_sites": scats_centroids,
        "scats_to_osm_node": scats_to_osm,
        "route_paths": route_paths,
        "background_road_segments": road_segments,
        "failed_edges": failed,
    }

    OUTPUT_JSON.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print()
    print("Saved:", OUTPUT_JSON)
    print("Route paths generated:", len(route_paths))
    print("Failed edges:", len(failed))

    if failed:
        print("First failed edges:", failed[:20])

    print("Done.")


if __name__ == "__main__":
    main()
