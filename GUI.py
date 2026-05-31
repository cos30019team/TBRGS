"""
COS30019 A2B TBRGS GUI, Boroondara SCATS Road Network Map.

This GUI does NOT use a static map image. It loads Boroondara SCATS Road Network Map, then draws the
roads, SCATS dots, labels, and route overlay with pygame.

Required files:
    GUI.py
    data/boroondara_exact_vector_map.json
    data/final_data_processing_output.csv
    GNN_model/GeneratedFiles/node_lookup_table.csv
    GNN_model/GeneratedFiles/adjacency_matrix.npy, optional

Run:
    python -u GUI.py
"""

from __future__ import annotations

import csv
import heapq
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Tuple, Optional, List

import pygame

try:
    import numpy as np
except Exception:
    np = None


# -----------------------------------------------------------------------------
# Constants and paths
# -----------------------------------------------------------------------------

WIDTH, HEIGHT = 1384, 865
FPS = 60
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
GNN_DIR = os.path.join(BASE_DIR, "GNN_model", "GeneratedFiles")
MODEL_ARTIFACTS_DIR = os.path.join(BASE_DIR, "models", "artifacts_person1")

EXACT_MAP_JSON = os.path.join(DATA_DIR, "boroondara_exact_vector_map.json")
PROCESSED_DATA = os.path.join(DATA_DIR, "final_data_processing_output.csv")
NODE_LOOKUP = os.path.join(GNN_DIR, "node_lookup_table.csv")
ADJ_MATRIX = os.path.join(GNN_DIR, "adjacency_matrix.npy")

LSTM_PREDICTIONS = os.path.join(MODEL_ARTIFACTS_DIR, "predictions_test.csv")
OSM_ROUTE_CACHE = os.path.join(DATA_DIR, "map_osm", "osm_route_cache.json")


# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------

pygame.init()
pygame.display.set_caption("TBRGS, Traffic Based Route Guidance System")
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
CLOCK = pygame.time.Clock()

FONT_TITLE = pygame.font.SysFont("Segoe UI", 30, bold=True)
FONT_H2 = pygame.font.SysFont("Segoe UI", 19, bold=True)
FONT_NORMAL = pygame.font.SysFont("Segoe UI", 14)
FONT_SMALL = pygame.font.SysFont("Segoe UI", 12)
FONT_TINY = pygame.font.SysFont("Segoe UI", 10)
FONT_BOLD = pygame.font.SysFont("Segoe UI", 14, bold=True)
FONT_BUTTON = pygame.font.SysFont("Segoe UI", 14, bold=True)
FONT_MAP_LABEL = pygame.font.SysFont("Segoe UI", 8)
FONT_MAP_LABEL_BOLD = pygame.font.SysFont("Segoe UI", 8, bold=True)

LIGHT = {
    "name": "Light",
    "bg": (245, 247, 251),
    "panel": (255, 255, 255),
    "panel2": (238, 243, 250),
    "input": (252, 253, 255),
    "text": (30, 38, 52),
    "muted": (125, 139, 160),
    "line": (201, 214, 235),
    "primary": (52, 102, 240),
    "primary_hover": (40, 86, 220),
    "green": (34, 160, 95),
    "danger": (220, 80, 85),
    "warning": (218, 142, 32),
    "map_bg": (250, 252, 255),
    "road": (30, 35, 40),
    "road_light": (150, 155, 160),
    "freeway": (76, 180, 56),
    "scats_ref": (218, 112, 198),
    "scats_data": (40, 110, 245),
    "route": (28, 166, 96),
}

DARK = {
    "name": "Dark",
    "bg": (18, 22, 31),
    "panel": (28, 34, 47),
    "panel2": (38, 45, 62),
    "input": (35, 42, 56),
    "text": (235, 239, 247),
    "muted": (160, 170, 190),
    "line": (62, 72, 92),
    "primary": (95, 150, 255),
    "primary_hover": (118, 166, 255),
    "green": (70, 205, 145),
    "danger": (235, 95, 95),
    "warning": (240, 180, 80),
    "map_bg": (22, 27, 38),
    "road": (210, 216, 225),
    "road_light": (105, 115, 135),
    "freeway": (90, 210, 95),
    "scats_ref": (230, 125, 220),
    "scats_data": (105, 165, 255),
    "route": (70, 225, 150),
}


def rounded(surface, rect, color, radius=14, border=None, width=1):
    pygame.draw.rect(surface, color, rect, border_radius=radius)
    if border:
        pygame.draw.rect(surface, border, rect, width=width, border_radius=radius)


def text(surface, value, font, color, x, y, max_width=None):
    value = str(value)
    if max_width is not None:
        while font.size(value)[0] > max_width and len(value) > 4:
            value = value[:-4] + "..."
    img = font.render(value, True, color)
    surface.blit(img, (int(x), int(y)))
    return img.get_rect(topleft=(int(x), int(y)))


def wrap_text(surface, value, font, color, x, y, max_width, line_gap=5):
    words = str(value).split()
    line = ""
    cur_y = y
    for word in words:
        test = word if not line else line + " " + word
        if font.size(test)[0] <= max_width:
            line = test
        else:
            text(surface, line, font, color, x, cur_y)
            cur_y += font.get_height() + line_gap
            line = word
    if line:
        text(surface, line, font, color, x, cur_y)
        cur_y += font.get_height() + line_gap
    return cur_y


def valid_time(t: str) -> bool:
    try:
        h, m = t.strip().split(":")
        return 0 <= int(h) <= 23 and 0 <= int(m) <= 59
    except Exception:
        return False


def is_dark_time(t: str) -> bool:
    if not valid_time(t):
        h = datetime.now().hour
    else:
        h = int(t.split(":")[0])
    return h >= 19 or h < 6


def clean_scats(value) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if not s:
        return ""
    try:
        return str(int(float(s)))
    except Exception:
        return s


# -----------------------------------------------------------------------------
# Widgets
# -----------------------------------------------------------------------------

class TextInput:
    def __init__(self, rect, label, value="", placeholder=""):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.value = value
        self.placeholder = placeholder
        self.active = False
        self.cursor = 0

    def handle(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.active = self.rect.collidepoint(event.pos)
        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_BACKSPACE:
                self.value = self.value[:-1]
            elif event.key == pygame.K_DELETE:
                self.value = ""
            elif event.key in (pygame.K_RETURN, pygame.K_TAB):
                self.active = False
            elif event.unicode and len(self.value) < 40:
                self.value += event.unicode

    def draw(self, surface, theme):
        text(surface, self.label, FONT_SMALL, theme["muted"], self.rect.x, self.rect.y - 18)
        rounded(surface, self.rect, theme["input"], 10, theme["primary"] if self.active else theme["line"], 2 if self.active else 1)
        shown = self.value if self.value else self.placeholder
        col = theme["text"] if self.value else theme["muted"]
        text(surface, shown, FONT_NORMAL, col, self.rect.x + 12, self.rect.y + 12, self.rect.width - 24)
        if self.active:
            self.cursor += 1
            if (self.cursor // 30) % 2 == 0:
                tw = FONT_NORMAL.size(self.value)[0]
                cx = min(self.rect.x + 12 + tw + 2, self.rect.right - 10)
                pygame.draw.line(surface, theme["text"], (cx, self.rect.y + 10), (cx, self.rect.y + self.rect.height - 10), 1)


class Button:
    def __init__(self, rect, label, callback, kind="primary"):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.callback = callback
        self.kind = kind
        self.hover = False

    def handle(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(event.pos)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.rect.collidepoint(event.pos):
            self.callback()

    def draw(self, surface, theme):
        if self.kind == "primary":
            bg = theme["primary_hover"] if self.hover else theme["primary"]
            fg = (255, 255, 255)
            border = None
        else:
            bg = theme["panel2"]
            fg = theme["text"]
            border = theme["line"]
        rounded(surface, self.rect, bg, 12, border, 1)
        img = FONT_BUTTON.render(self.label, True, fg)
        surface.blit(img, img.get_rect(center=self.rect.center))


class Toggle:
    def __init__(self, rect, label, value=True, callback=None):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.value = value
        self.callback = callback

    def handle(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.rect.collidepoint(event.pos):
            self.value = not self.value
            if self.callback:
                self.callback(self.value)

    def draw(self, surface, theme):
        text(surface, self.label, FONT_NORMAL, theme["text"], self.rect.x, self.rect.y + 4)
        tx = self.rect.right - 54
        ty = self.rect.y + 2
        pygame.draw.rect(surface, theme["green"] if self.value else theme["line"], (tx, ty, 48, 24), border_radius=12)
        kx = tx + 26 if self.value else tx + 4
        pygame.draw.circle(surface, (255,255,255), (kx + 8, ty + 12), 9)


class Dropdown:
    def __init__(self, rect, label, options, selected=None):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.options = options
        self.selected = selected or options[0]
        self.open = False

    def handle(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.open = not self.open
            elif self.open:
                chosen = False
                for i, opt in enumerate(self.options):
                    r = pygame.Rect(self.rect.x, self.rect.bottom + i * 36, self.rect.width, 36)
                    if r.collidepoint(event.pos):
                        self.selected = opt
                        self.open = False
                        chosen = True
                        break
                if not chosen:
                    self.open = False

    def draw(self, surface, theme):
        text(surface, self.label, FONT_SMALL, theme["muted"], self.rect.x, self.rect.y - 18)
        rounded(surface, self.rect, theme["input"], 10, theme["line"], 1)
        text(surface, self.selected, FONT_NORMAL, theme["text"], self.rect.x + 12, self.rect.y + 12, self.rect.width - 45)
        text(surface, "▼", FONT_NORMAL, theme["muted"], self.rect.right - 28, self.rect.y + 12)
        if self.open:
            for i, opt in enumerate(self.options):
                r = pygame.Rect(self.rect.x, self.rect.bottom + i * 36, self.rect.width, 36)
                rounded(surface, r, theme["panel"], 8, theme["line"], 1)
                text(surface, opt, FONT_NORMAL, theme["text"], r.x + 12, r.y + 10, r.width - 24)


# -----------------------------------------------------------------------------
# Data and route logic
# -----------------------------------------------------------------------------

@dataclass
class MapNode:
    scats: str
    x: float
    y: float
    label_x: float
    label_y: float
    data_site: bool = False


class ExactMapRepository:
    def __init__(self):
        self.page_width = 841.49
        self.page_height = 595.50
        self.segments: List[list] = []
        self.nodes: Dict[str, MapNode] = {}
        self.data_sites: set[str] = set()
        self.graph: Dict[str, List[Tuple[str, float]]] = {}
        self.flow_cache: Optional[Dict[Tuple[str, int], float]] = None
        self.lstm_prediction_cache: Optional[Dict[Tuple[str, int], float]] = None
        self.gru_prediction_cache: Optional[Dict[Tuple[str, int], float]] = None
        self.last_prediction_source = "Historical Avg"
        self.lstm_hits = 0
        self.gru_hits = 0
        self.historical_hits = 0

        # OSM road-following map cache.
        self.osm_cache = {}
        self.osm_scats_sites = {}
        self.osm_route_paths = {}
        self.osm_road_segments = []
        self.has_osm_map = False
        self.geo_bounds = None

        self.load_all()

    def load_all(self):
        self.load_exact_pdf_vector_map()
        self.load_data_sites_from_node_lookup()
        self.build_route_graph()
        self.load_osm_route_cache()

    def load_exact_pdf_vector_map(self):
        if not os.path.exists(EXACT_MAP_JSON):
            raise FileNotFoundError(
                "Missing data/boroondara_exact_vector_map.json. Put the generated vector map JSON inside the data folder."
            )
        with open(EXACT_MAP_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.page_width = float(data.get("page_width", self.page_width))
        self.page_height = float(data.get("page_height", self.page_height))
        self.segments = data.get("segments", [])
        for sid, item in data.get("scats", {}).items():
            self.nodes[clean_scats(sid)] = MapNode(
                scats=clean_scats(sid),
                x=float(item["x"]),
                y=float(item["y"]),
                label_x=float(item.get("label_x", item["x"] + 5)),
                label_y=float(item.get("label_y", item["y"] - 5)),
                data_site=False,
            )

    def load_data_sites_from_node_lookup(self):
        if not os.path.exists(NODE_LOOKUP):
            # Fallback to all map nodes if the GNN lookup is missing.
            self.data_sites = set(self.nodes.keys())
        else:
            with open(NODE_LOOKUP, "r", newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sid = clean_scats(row.get("SCATS Number"))
                    if sid in self.nodes:
                        self.data_sites.add(sid)
        for sid in self.data_sites:
            if sid in self.nodes:
                self.nodes[sid].data_site = True


    def load_osm_route_cache(self):
        """
        Loads road-following OpenStreetMap geometry generated by:
        tools/build_osm_route_cache.py

        This keeps the assignment SCATS graph and routing logic, but draws
        route paths using real road geometry instead of straight node-to-node lines.
        """
        if not os.path.exists(OSM_ROUTE_CACHE):
            self.has_osm_map = False
            return

        try:
            with open(OSM_ROUTE_CACHE, "r", encoding="utf-8") as f:
                self.osm_cache = json.load(f)

            self.osm_scats_sites = self.osm_cache.get("scats_sites", {})
            self.osm_route_paths = self.osm_cache.get("route_paths", {})
            self.osm_road_segments = self.osm_cache.get("background_road_segments", [])

            lats = []
            lons = []

            for site, item in self.osm_scats_sites.items():
                try:
                    lat = float(item["lat"])
                    lon = float(item["lon"])
                except Exception:
                    continue

                if lat != 0 and lon != 0:
                    lats.append(lat)
                    lons.append(lon)

            for segment in self.osm_road_segments:
                for point in segment:
                    try:
                        lat = float(point[0])
                        lon = float(point[1])
                    except Exception:
                        continue

                    if lat != 0 and lon != 0:
                        lats.append(lat)
                        lons.append(lon)

            if lats and lons:
                pad_lat = (max(lats) - min(lats)) * 0.04 or 0.005
                pad_lon = (max(lons) - min(lons)) * 0.04 or 0.005
                self.geo_bounds = {
                    "min_lat": min(lats) - pad_lat,
                    "max_lat": max(lats) + pad_lat,
                    "min_lon": min(lons) - pad_lon,
                    "max_lon": max(lons) + pad_lon,
                }
                self.has_osm_map = True
            else:
                self.has_osm_map = False

        except Exception as exc:
            print("Could not load OSM route cache:", exc)
            self.has_osm_map = False

    def get_site_geo(self, sid: str):
        """
        Returns (lat, lon) for a SCATS site from the OSM route cache.
        """
        sid = clean_scats(sid)
        item = self.osm_scats_sites.get(sid)

        if not item:
            return None

        try:
            lat = float(item["lat"])
            lon = float(item["lon"])
        except Exception:
            return None

        if lat == 0 or lon == 0:
            return None

        return lat, lon

    def get_osm_route_coords(self, src: str, dst: str):
        """
        Returns road-following [lat, lon] coordinates for src -> dst.
        Tries both directions because the assignment graph is treated as undirected.
        """
        src = clean_scats(src)
        dst = clean_scats(dst)

        forward = self.osm_route_paths.get(f"{src}->{dst}")
        if forward and forward.get("coords"):
            return forward["coords"]

        reverse = self.osm_route_paths.get(f"{dst}->{src}")
        if reverse and reverse.get("coords"):
            coords = list(reverse["coords"])
            coords.reverse()
            return coords

        return None

    def selected_path_osm_coords(self, selected_path: List[str]):
        """
        Converts a selected SCATS route into one continuous road-following
        coordinate list.
        """
        if not selected_path or len(selected_path) < 2:
            return []

        coords = []

        for src, dst in zip(selected_path[:-1], selected_path[1:]):
            segment = self.get_osm_route_coords(src, dst)

            if segment:
                if coords:
                    coords.extend(segment[1:])
                else:
                    coords.extend(segment)
            else:
                # Fallback only for unexpected missing edges.
                a = self.get_site_geo(src)
                b = self.get_site_geo(dst)
                if a and b:
                    fallback = [[a[0], a[1]], [b[0], b[1]]]
                    if coords:
                        coords.extend(fallback[1:])
                    else:
                        coords.extend(fallback)

        return coords


    def distance_km(self, a: str, b: str) -> float:
        # The PDF scale bar is 4 km. The actual map width covers roughly 12 km west-east.
        # This is enough for assignment travel time estimation because the objective is to
        # use predicted flow + graph search, not replace Google Maps.
        na, nb = self.nodes[a], self.nodes[b]
        page_dist = math.hypot(na.x - nb.x, na.y - nb.y)
        km_per_pdf_unit = 12.0 / self.page_width
        return max(0.05, page_dist * km_per_pdf_unit)

    def build_route_graph(self):
        sites = sorted([s for s in self.data_sites if s in self.nodes])
        self.graph = {s: [] for s in sites}
        if len(sites) < 2:
            return

        # Candidate edges from exact PDF coordinates, kept local to avoid ugly cross-map lines.
        candidates = []
        for i, a in enumerate(sites):
            for b in sites[i+1:]:
                d_pdf = math.hypot(self.nodes[a].x - self.nodes[b].x, self.nodes[a].y - self.nodes[b].y)
                d_km = self.distance_km(a, b)
                candidates.append((d_pdf, d_km, a, b))
        candidates.sort()

        # 1) Minimum spanning tree guarantees every dataset SCATS site is reachable.
        parent = {s: s for s in sites}
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a,b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return False
            parent[rb] = ra
            return True
        added = set()
        for d_pdf, d_km, a, b in candidates:
            if union(a,b):
                self._add_edge(a,b,d_km,added)

        # 2) Add local neighbours for route alternatives and top-k paths.
        for a in sites:
            local = []
            for b in sites:
                if a == b:
                    continue
                d_pdf = math.hypot(self.nodes[a].x - self.nodes[b].x, self.nodes[a].y - self.nodes[b].y)
                local.append((d_pdf, b))
            local.sort()
            for d_pdf, b in local[:5]:
                # still local, but allows enough alternate routes
                self._add_edge(a, b, self.distance_km(a,b), added)

        # 3) If adjacency_matrix.npy is available, use it as additional candidate evidence.
        # It is not drawn directly because it is dense, but it helps route connectivity.
        self._merge_gnn_adjacency_edges(added)

    def _add_edge(self, a: str, b: str, km: float, added: set):
        key = tuple(sorted((a,b)))
        if key in added:
            return
        added.add(key)
        self.graph.setdefault(a, []).append((b, km))
        self.graph.setdefault(b, []).append((a, km))

    def _merge_gnn_adjacency_edges(self, added: set):
        if np is None or not os.path.exists(ADJ_MATRIX) or not os.path.exists(NODE_LOOKUP):
            return
        try:
            lookup_rows = []
            with open(NODE_LOOKUP, "r", newline="", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    lookup_rows.append((int(row["Node_ID"]), clean_scats(row["SCATS Number"])))
            node_to_scats = {nid: scats for nid, scats in lookup_rows if scats in self.data_sites and scats in self.nodes}
            A = np.load(ADJ_MATRIX)
            rows, cols = np.nonzero(A > 0)
            candidate_pairs = set()
            for i, j in zip(rows.tolist(), cols.tolist()):
                a = node_to_scats.get(i)
                b = node_to_scats.get(j)
                if not a or not b or a == b:
                    continue
                d_pdf = math.hypot(self.nodes[a].x - self.nodes[b].x, self.nodes[a].y - self.nodes[b].y)
                if d_pdf <= 95:  # avoid huge cross-map edges
                    candidate_pairs.add(tuple(sorted((a,b))))
            # Add only closest GNN-supported links so the route graph is useful but not noisy.
            sorted_pairs = sorted(candidate_pairs, key=lambda p: self.distance_km(p[0],p[1]))
            for a,b in sorted_pairs[:120]:
                self._add_edge(a,b,self.distance_km(a,b),added)
        except Exception:
            return

        def load_flow_cache(self):
            """
            Loads historical average hourly traffic flow from data/final_data_processing_output.csv.
            Cache key is (SCATS site, hour).
            """
        if self.flow_cache is not None:
            return

        totals: Dict[Tuple[str, int], List[float]] = {}

        if not os.path.exists(PROCESSED_DATA):
            self.flow_cache = {}
            return

        with open(PROCESSED_DATA, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)

            for row in reader:
                sid = clean_scats(row.get("SCATS Number"))

                if sid not in self.data_sites:
                    continue

                try:
                    hour = int(float(row.get("Hour", 0)))
                    vol = float(row.get("Hourly_Volume") or row.get("Volume") or 0)
                except Exception:
                    continue

                totals.setdefault((sid, hour), []).append(vol)

        self.flow_cache = {
            key: sum(values) / len(values)
            for key, values in totals.items()
            if values
        }


    def load_flow_cache(self):
        """
        Loads historical average hourly traffic flow from data/final_data_processing_output.csv.
        Cache key is (SCATS site, hour).
        """
        if self.flow_cache is not None:
            return

        totals = {}

        if not os.path.exists(PROCESSED_DATA):
            self.flow_cache = {}
            return

        with open(PROCESSED_DATA, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)

            for row in reader:
                sid = clean_scats(row.get("SCATS Number"))

                if sid not in self.data_sites:
                    continue

                try:
                    hour = int(float(row.get("Hour", 0)))
                    vol = float(row.get("Hourly_Volume") or row.get("Volume") or 0)
                except Exception:
                    continue

                totals.setdefault((sid, hour), []).append(vol)

        self.flow_cache = {
            key: sum(values) / len(values)
            for key, values in totals.items()
            if values
        }

    def load_lstm_prediction_cache(self):
        """
        Loads saved LSTM 15-minute predictions from:
        models/artifacts_person1/predictions_test.csv

        step_index_0to95 represents 15-minute intervals.
        Hour is calculated as step_index // 4.

        The route engine needs hourly flow:
        hourly_flow = pred_lstm_veh_15min * 4
        """
        if self.lstm_prediction_cache is not None:
            return

        predictions = {}

        if not os.path.exists(LSTM_PREDICTIONS):
            print("LSTM prediction file not found:", LSTM_PREDICTIONS)
            self.lstm_prediction_cache = {}
            return

        try:
            with open(LSTM_PREDICTIONS, "r", newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)

                for row in reader:
                    sid = clean_scats(row.get("scats_site", ""))

                    try:
                        step_index = int(float(row.get("step_index_0to95", 0)))
                        hour = max(0, min(23, step_index // 4))
                        pred_15min = float(row.get("pred_lstm_veh_15min", 0))
                        pred_hourly = max(0.0, pred_15min * 4.0)
                    except Exception:
                        continue

                    predictions.setdefault((sid, hour), []).append(pred_hourly)

            self.lstm_prediction_cache = {}

            for key, values in predictions.items():
                if values:
                    self.lstm_prediction_cache[key] = sum(values) / len(values)

            # LSTM prediction cache loaded successfully.

        except Exception as exc:
            print("Could not load LSTM prediction cache:", exc)
            self.lstm_prediction_cache = {}


    def load_gru_prediction_cache(self):
        """
        Loads saved GRU 15-minute predictions from:
        models/artifacts_person1/predictions_test.csv

        The prediction file uses step_index_0to95, where every step is 15 minutes.
        Hour is calculated as step_index // 4.

        The route engine needs hourly flow:
        hourly_flow = pred_gru_veh_15min * 4
        """
        if self.gru_prediction_cache is not None:
            return

        predictions = {}

        if not os.path.exists(LSTM_PREDICTIONS):
            print("Prediction file not found:", LSTM_PREDICTIONS)
            self.gru_prediction_cache = {}
            return

        try:
            with open(LSTM_PREDICTIONS, "r", newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)

                for row in reader:
                    sid = clean_scats(row.get("scats_site", ""))

                    if "pred_gru_veh_15min" not in row:
                        continue

                    try:
                        step_index = int(float(row.get("step_index_0to95", 0)))
                        hour = max(0, min(23, step_index // 4))
                        pred_15min = float(row.get("pred_gru_veh_15min", 0))
                        pred_hourly = max(0.0, pred_15min * 4.0)
                    except Exception:
                        continue

                    predictions.setdefault((sid, hour), []).append(pred_hourly)

            self.gru_prediction_cache = {}

            for key, values in predictions.items():
                if values:
                    self.gru_prediction_cache[key] = sum(values) / len(values)

        except Exception as exc:
            print("Could not load GRU prediction cache:", exc)
            self.gru_prediction_cache = {}


    def flow_to_speed(self, flow: float) -> float:
        """
        Converts hourly traffic flow to speed using the assignment formula:
        flow = -1.4648375 * speed^2 + 93.75 * speed

        Uses the under-capacity branch and caps speed at 60 km/h.
        """
        a = -1.4648375
        b = 93.75
        c = -max(0.0, float(flow))

        disc = b * b - 4 * a * c

        if disc < 0:
            return 32.0

        root1 = (-b + math.sqrt(disc)) / (2 * a)
        root2 = (-b - math.sqrt(disc)) / (2 * a)

        candidates = [r for r in (root1, root2) if r > 0]
        speed = max(candidates) if candidates else 32.0

        return max(5.0, min(60.0, speed))


    def predict_flow(self, scats: str, date_text: str, time_text: str, model_name: str) -> float:
        """
        Predicts hourly traffic flow for a SCATS site.

        Best Available:
        - tries GRU first
        - then LSTM
        - then Historical Avg

        LSTM:
        - uses pred_lstm_veh_15min when available

        GRU:
        - uses pred_gru_veh_15min when available

        Custom:
        - currently falls back to historical average until custom output is supplied
        """
        self.load_flow_cache()

        hour = int(time_text.split(":")[0]) if valid_time(time_text) else 8
        model = str(model_name or "").strip().lower()

        def historical_fallback() -> float:
            self.historical_hits += 1
            self.last_prediction_source = "Historical Avg"

            if self.flow_cache:
                return float(
                    self.flow_cache.get(
                        (scats, hour),
                        self.flow_cache.get((scats, 8), 600.0)
                    )
                )

            return 600.0

        def try_gru() -> Optional[float]:
            try:
                self.load_gru_prediction_cache()

                if self.gru_prediction_cache:
                    value = self.gru_prediction_cache.get((scats, hour))

                    if value is None:
                        value = self.gru_prediction_cache.get((scats, 8))

                    if value is not None:
                        self.gru_hits += 1
                        self.last_prediction_source = "GRU"
                        return float(value)

            except Exception as exc:
                print("GRU prediction failed, using fallback instead:", exc)

            return None

        def try_lstm() -> Optional[float]:
            try:
                self.load_lstm_prediction_cache()

                if self.lstm_prediction_cache:
                    value = self.lstm_prediction_cache.get((scats, hour))

                    if value is None:
                        value = self.lstm_prediction_cache.get((scats, 8))

                    if value is not None:
                        self.lstm_hits += 1
                        self.last_prediction_source = "LSTM"
                        return float(value)

            except Exception as exc:
                print("LSTM prediction failed, using fallback instead:", exc)

            return None

        if model == "gru":
            value = try_gru()
            if value is not None:
                return value

        elif model == "lstm":
            value = try_lstm()
            if value is not None:
                return value

        elif model == "best available":
            value = try_gru()
            if value is not None:
                return value

            value = try_lstm()
            if value is not None:
                return value

        # Custom and Historical Avg currently use historical fallback.
        return historical_fallback()

    def edge_time_minutes(self, src: str, dst: str, km: float, date_text: str, time_text: str, model: str) -> Tuple[float, float]:
        flow = self.predict_flow(src, date_text, time_text, model)
        speed = self.flow_to_speed(flow)
        travel = (km / speed) * 60.0
        delay = 0.5  # 30 seconds per controlled intersection
        return travel + delay, flow

    def top_k_routes(self, origin: str, dest: str, date_text: str, time_text: str, model: str, k: int) -> List[dict]:
        self.lstm_hits = 0
        self.gru_hits = 0
        self.historical_hits = 0
        self.last_prediction_source = "Historical Avg"
        origin = clean_scats(origin)
        dest = clean_scats(dest)
        if origin not in self.graph or dest not in self.graph:
            return []
        heap = [(0.0, origin, [origin], 0.0, [])]
        results = []
        seen_paths = set()
        limit = 2500
        while heap and len(results) < k and limit > 0:
            limit -= 1
            cost, node, path, distance_sum, flows = heapq.heappop(heap)
            if node == dest:
                key = tuple(path)
                if key not in seen_paths:
                    seen_paths.add(key)
                    results.append({
                        "path": path,
                        "estimated_time_min": round(cost, 2),
                        "distance_km": round(distance_sum, 2),
                        "avg_flow": round(sum(flows)/len(flows), 1) if flows else 0.0,
                    })
                continue
            if len(path) > min(18, len(self.graph)):
                continue
            for nb, km in self.graph.get(node, []):
                if nb in path:
                    continue
                tmin, flow = self.edge_time_minutes(node, nb, km, date_text, time_text, model)
                heapq.heappush(heap, (cost + tmin, nb, path + [nb], distance_sum + km, flows + [flow]))
        return results


# -----------------------------------------------------------------------------
# Map renderer
# -----------------------------------------------------------------------------


class ExactVectorMapView:
    def __init__(self, rect: pygame.Rect, repo: ExactMapRepository):
        self.rect = rect
        self.repo = repo
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.dragging = False
        self.last_mouse = (0, 0)
        self.hover_node: Optional[str] = None

    def reset_view(self):
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0

    def pdf_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        nx = (x / self.repo.page_width) * self.rect.width
        ny = (y / self.repo.page_height) * self.rect.height
        sx = self.rect.x + self.rect.width / 2 + (nx - self.rect.width / 2) * self.zoom + self.pan_x
        sy = self.rect.y + self.rect.height / 2 + (ny - self.rect.height / 2) * self.zoom + self.pan_y
        return int(sx), int(sy)

    def geo_to_screen(self, lat: float, lon: float) -> Tuple[int, int]:
        """
        Converts real latitude/longitude to the map panel.
        Used when the OSM route cache is available.
        """
        b = self.repo.geo_bounds

        if not b:
            return self.pdf_to_screen(0, 0)

        min_lat = float(b["min_lat"])
        max_lat = float(b["max_lat"])
        min_lon = float(b["min_lon"])
        max_lon = float(b["max_lon"])

        x_norm = (float(lon) - min_lon) / max(0.000001, (max_lon - min_lon))
        y_norm = (max_lat - float(lat)) / max(0.000001, (max_lat - min_lat))

        sx_base = self.rect.x + x_norm * self.rect.width
        sy_base = self.rect.y + y_norm * self.rect.height

        sx = self.rect.x + self.rect.width / 2 + (sx_base - (self.rect.x + self.rect.width / 2)) * self.zoom + self.pan_x
        sy = self.rect.y + self.rect.height / 2 + (sy_base - (self.rect.y + self.rect.height / 2)) * self.zoom + self.pan_y

        return int(sx), int(sy)

    def site_to_screen(self, sid: str) -> Optional[Tuple[int, int]]:
        """
        Converts a SCATS site to screen coordinates.
        Uses real OSM lat/lon when available, otherwise falls back to PDF coordinates.
        """
        if self.repo.has_osm_map:
            geo = self.repo.get_site_geo(sid)
            if geo:
                return self.geo_to_screen(geo[0], geo[1])

        n = self.repo.nodes.get(sid)
        if n:
            return self.pdf_to_screen(n.x, n.y)

        return None

    def inside(self, pos):
        return self.rect.collidepoint(pos)

    def handle(self, event):
        if event.type == pygame.MOUSEWHEEL:
            mx, my = pygame.mouse.get_pos()

            if self.inside((mx, my)):
                old = self.zoom
                self.zoom = max(0.65, min(4.5, self.zoom * (1.12 if event.y > 0 else 0.90)))
                self.pan_x = mx - (mx - self.pan_x - self.rect.centerx) * (self.zoom / old) - self.rect.centerx
                self.pan_y = my - (my - self.pan_y - self.rect.centery) * (self.zoom / old) - self.rect.centery

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.inside(event.pos):
            self.dragging = True
            self.last_mouse = event.pos

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False

        elif event.type == pygame.MOUSEMOTION:
            self.hover_node = self.node_at(event.pos)

            if self.dragging:
                dx = event.pos[0] - self.last_mouse[0]
                dy = event.pos[1] - self.last_mouse[1]
                self.pan_x += dx
                self.pan_y += dy
                self.last_mouse = event.pos

    def node_at(self, pos) -> Optional[str]:
        if not self.inside(pos):
            return None

        best_sid = None
        best_d = 9999

        for sid, n in self.repo.nodes.items():
            if not n.data_site:
                continue

            pt = self.site_to_screen(sid)
            if not pt:
                continue

            sx, sy = pt
            d = math.hypot(pos[0] - sx, pos[1] - sy)

            if d < best_d and d <= 10:
                best_d = d
                best_sid = sid

        return best_sid


    def draw_pdf_background(self, surface, theme):
        """
        Fallback original PDF/vector background.
        """
        for x1, y1, x2, y2, c, w in self.repo.segments:
            p1 = self.pdf_to_screen(x1, y1)
            p2 = self.pdf_to_screen(x2, y2)

            if c == "green":
                col = theme["freeway"]
                width = max(1, int(w * self.zoom * 1.4))
            elif c == "grey":
                col = theme["road_light"]
                width = max(1, int(w * self.zoom))
            else:
                col = theme["road"]
                width = max(1, int(w * self.zoom))

            pygame.draw.line(surface, col, p1, p2, width)


    def draw_osm_background(self, surface, theme):
        """
        Draws OpenStreetMap drive-network roads.

        Balanced mode:
        - zoomed out: draw every second road segment
        - zoomed in: draw all road segments
        The selected route always draws fully.
        """
        if self.zoom < 1.2:
            step = 2
            width = 1
        elif self.zoom < 2.0:
            step = 1
            width = 1
        else:
            step = 1
            width = max(1, int(1.3 * self.zoom))

        for index, segment in enumerate(self.repo.osm_road_segments):
            if index % step != 0:
                continue

            pts = []

            for point in segment:
                try:
                    lat = float(point[0])
                    lon = float(point[1])
                except Exception:
                    continue

                sx, sy = self.geo_to_screen(lat, lon)

                # Skip segments far outside the map panel.
                if (
                    sx < self.rect.x - 100 or sx > self.rect.right + 100 or
                    sy < self.rect.y - 100 or sy > self.rect.bottom + 100
                ):
                    continue

                pts.append((sx, sy))

            if len(pts) > 1:
                pygame.draw.lines(surface, theme["road_light"], False, pts, width)


    def draw_selected_route(self, surface, theme, selected_path: Optional[List[str]]):
        """
        Draws selected route. Uses real OSM road-following geometry when available.
        """
        if not selected_path or len(selected_path) < 2:
            return

        if self.repo.has_osm_map:
            coords = self.repo.selected_path_osm_coords(selected_path)
            pts = []

            for point in coords:
                try:
                    lat = float(point[0])
                    lon = float(point[1])
                except Exception:
                    continue

                pts.append(self.geo_to_screen(lat, lon))

            if len(pts) > 1:
                pygame.draw.lines(surface, theme["route"], False, pts, max(4, int(4 * self.zoom)))

            for sid in selected_path:
                pt = self.site_to_screen(sid)
                if pt:
                    pygame.draw.circle(surface, theme["route"], pt, 6)

            return

        # Fallback original straight-line drawing.
        pts = []

        for sid in selected_path:
            n = self.repo.nodes.get(sid)
            if n:
                pts.append(self.pdf_to_screen(n.x, n.y))

        if len(pts) > 1:
            pygame.draw.lines(surface, theme["route"], False, pts, 5)

            for p in pts:
                pygame.draw.circle(surface, theme["route"], p, 5)

    def draw(self, surface, theme, selected_path: Optional[List[str]], origin: str, dest: str):
        rounded(surface, self.rect, theme["map_bg"], 10, theme["line"], 1)
        old_clip = surface.get_clip()
        surface.set_clip(self.rect)

        if self.repo.has_osm_map:
            self.draw_osm_background(surface, theme)
        else:
            self.draw_pdf_background(surface, theme)

        # Draw selected route overlay on top of roads.
        self.draw_selected_route(surface, theme, selected_path)

        # Draw SCATS dots and labels.
        for sid, n in self.repo.nodes.items():
            pt = self.site_to_screen(sid)

            if not pt:
                continue

            sx, sy = pt

            if sx < self.rect.x - 20 or sx > self.rect.right + 20 or sy < self.rect.y - 20 or sy > self.rect.bottom + 20:
                continue

            is_origin = sid == origin
            is_dest = sid == dest
            is_route = selected_path and sid in selected_path

            if is_origin or is_dest:
                col = theme["warning"]
                r = 7
            elif is_route:
                col = theme["route"]
                r = 6
            elif n.data_site:
                col = theme["scats_data"]
                r = 5
            else:
                col = theme["scats_ref"]
                r = 3

            pygame.draw.circle(surface, col, (sx, sy), r)
            pygame.draw.circle(surface, theme["panel"], (sx, sy), max(1, r - 3))

            label_font = FONT_MAP_LABEL_BOLD if n.data_site or is_origin or is_dest else FONT_MAP_LABEL
            label_col = theme["text"] if n.data_site or is_origin or is_dest else theme["muted"]

            text(surface, sid, label_font, label_col, sx + 6, sy - 8)

            if is_origin:
                text(surface, "O", FONT_BOLD, theme["warning"], sx + 8, sy + 4)

            if is_dest:
                text(surface, "D", FONT_BOLD, theme["warning"], sx + 8, sy + 18)

        surface.set_clip(old_clip)

        # Legend and hover outside clip.
        legend_y = self.rect.bottom - 27
        pygame.draw.circle(surface, theme["scats_ref"], (self.rect.x + 20, legend_y), 4)
        text(surface, "Reference SCATS", FONT_TINY, theme["muted"], self.rect.x + 28, legend_y - 7)

        pygame.draw.circle(surface, theme["scats_data"], (self.rect.x + 125, legend_y), 4)
        text(surface, "Dataset SCATS", FONT_TINY, theme["muted"], self.rect.x + 133, legend_y - 7)

        pygame.draw.line(surface, theme["road_light"], (self.rect.x + 250, legend_y), (self.rect.x + 290, legend_y), 3)
        text(surface, "OSM roads", FONT_TINY, theme["muted"], self.rect.x + 297, legend_y - 7)

        pygame.draw.line(surface, theme["route"], (self.rect.x + 395, legend_y), (self.rect.x + 435, legend_y), 4)
        text(surface, "Selected route", FONT_TINY, theme["muted"], self.rect.x + 442, legend_y - 7)

        if self.hover_node:
            text(surface, f"Hover: {self.hover_node}", FONT_SMALL, theme["text"], self.rect.right - 120, self.rect.bottom - 28)


# -----------------------------------------------------------------------------
# Main GUI
# -----------------------------------------------------------------------------

class TBRGSGUI:
    def __init__(self):
        self.repo = ExactMapRepository()
        self.auto_theme = True
        self.theme = DARK if is_dark_time("08:00") else LIGHT
        self.status = f"Loaded OSM road-following map: {len(self.repo.nodes)} SCATS labels, {len(self.repo.data_sites)} dataset sites." if self.repo.has_osm_map else f"Loaded Boroondara SCATS Road Network Map: {len(self.repo.nodes)} SCATS labels, {len(self.repo.data_sites)} dataset sites."
        self.status_type = "info"
        self.pick_mode: Optional[str] = None
        self.routes: List[dict] = []
        self.selected_route = 0

        self.origin = TextInput((18, 208, 155, 42), "Origin", "2000", "SCATS")
        self.dest = TextInput((193, 208, 155, 42), "Destination", "3002", "SCATS")
        self.date = TextInput((18, 286, 155, 42), "Prediction Date", "2006-10-01", "YYYY-MM-DD")
        self.time = TextInput((193, 286, 155, 42), "Prediction Time", "08:00", "HH:MM")
        self.model = Dropdown((18, 364, 155, 42), "Model", ["Best Available", "LSTM", "GRU", "Custom", "Historical Avg"], "Best Available")
        self.topk = Dropdown((193, 364, 155, 42), "Top-K Routes", ["1", "2", "3", "4", "5"], "5")
        self.toggle = Toggle((18, 438, 330, 30), "Auto theme by time", True, self.set_auto)

        self.buttons = [
            Button((18, 486, 155, 42), "Calculate", self.calculate, "primary"),
            Button((193, 486, 155, 42), "Switch Theme", self.switch_theme, "secondary"),
            Button((18, 542, 155, 38), "Fit Map", self.fit_map, "secondary"),
            Button((193, 542, 155, 38), "Reset", self.reset, "secondary"),
        ]
        self.inputs = [self.origin, self.dest, self.date, self.time]
        self.dropdowns = [self.model, self.topk]

        self.map_view = ExactVectorMapView(pygame.Rect(412, 184, 936, 464), self.repo)
        self.apply_auto_theme()

    def set_auto(self, value):
        self.auto_theme = value
        self.apply_auto_theme()
        self.status = "Auto theme enabled." if value else "Auto theme disabled. Use Switch Theme manually."

    def apply_auto_theme(self):
        if self.auto_theme:
            self.theme = DARK if is_dark_time(self.time.value) else LIGHT

    def switch_theme(self):
        if self.auto_theme:
            self.status = "Turn Auto Theme off before manually switching theme."
            self.status_type = "warning"
            return
        self.theme = DARK if self.theme["name"] == "Light" else LIGHT
        self.status = f"Theme switched to {self.theme['name']} mode."
        self.status_type = "info"

    def fit_map(self):
        self.map_view.reset_view()
        self.status = "Map view reset to the OSM road-following map extent." if self.repo.has_osm_map else "Map view reset to the exact PDF vector map extent."
        self.status_type = "info"

    def reset(self):
        self.origin.value = "2000"
        self.dest.value = "3002"
        self.date.value = "2006-10-01"
        self.time.value = "08:00"
        self.model.selected = "Best Available"
        self.topk.selected = "5"
        self.routes = []
        self.selected_route = 0
        self.pick_mode = None
        self.map_view.reset_view()
        self.apply_auto_theme()
        self.status = "Inputs reset."
        self.status_type = "info"

    def calculate(self):
        self.apply_auto_theme()
        o = clean_scats(self.origin.value)
        d = clean_scats(self.dest.value)
        if o not in self.repo.data_sites:
            self.status = f"Origin {o} is not one of the dataset SCATS sites from node_lookup_table.csv."
            self.status_type = "danger"
            return
        if d not in self.repo.data_sites:
            self.status = f"Destination {d} is not one of the dataset SCATS sites from node_lookup_table.csv."
            self.status_type = "danger"
            return
        if not valid_time(self.time.value):
            self.status = "Prediction time must be HH:MM, for example 08:00 or 17:30."
            self.status_type = "danger"
            return
        self.status = "Calculating routes using exact map positions, traffic flow conversion, and 30 second intersection delay..."
        pygame.display.flip()
        self.routes = self.repo.top_k_routes(o, d, self.date.value, self.time.value, self.model.selected, int(self.topk.selected))
        self.selected_route = 0
        if self.routes:
            parts = []

            if self.repo.gru_hits > 0:
                parts.append(f"{self.repo.gru_hits} GRU")

            if self.repo.lstm_hits > 0:
                parts.append(f"{self.repo.lstm_hits} LSTM")

            if self.repo.historical_hits > 0:
                parts.append(f"{self.repo.historical_hits} Historical Avg fallback")

            if parts:
                source = " + ".join(parts)
            else:
                source = "Historical Avg"

            self.status = f"Found {len(self.routes)} route(s) using {source} traffic flow. Select a result card to highlight it on the road network map."
            self.status_type = "success"
        else:
            self.status = "No route found. Check origin/destination or route graph connectivity."
            self.status_type = "danger"

    def handle_event(self, event):
        self.map_view.handle(event)
        for inp in self.inputs:
            inp.handle(event)
        for dd in self.dropdowns:
            dd.handle(event)
        self.toggle.handle(event)
        for btn in self.buttons:
            btn.handle(event)

        if event.type == pygame.KEYDOWN and self.time.active:
            self.apply_auto_theme()

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.origin.rect.collidepoint(event.pos):
                self.pick_mode = "origin"
                self.status = "Origin box selected. Click a blue dataset SCATS node on the map."
                self.status_type = "info"
            elif self.dest.rect.collidepoint(event.pos):
                self.pick_mode = "dest"
                self.status = "Destination box selected. Click a blue dataset SCATS node on the map."
                self.status_type = "info"
            elif self.pick_mode and self.map_view.hover_node:
                if self.pick_mode == "origin":
                    self.origin.value = self.map_view.hover_node
                    self.status = f"Origin set to {self.map_view.hover_node}."
                else:
                    self.dest.value = self.map_view.hover_node
                    self.status = f"Destination set to {self.map_view.hover_node}."
                self.status_type = "success"
                self.pick_mode = None
            self.handle_route_click(event.pos)

    def handle_route_click(self, pos):
        if not self.routes:
            return
        y = 724
        for i, _ in enumerate(self.routes[:5]):
            r = pygame.Rect(412 + i * 182, y, 170, 88)
            if r.collidepoint(pos):
                self.selected_route = i
                break

    def draw_header(self, surface, theme):
        text(surface, "TBRGS", FONT_TITLE, theme["text"], 18, 50)
        text(surface, "Traffic Based Route Guidance System, COS30019 A2B", FONT_NORMAL, theme["muted"], 116, 61)
        now = datetime.now().strftime("%d %b %Y, %I:%M %p")
        text(surface, now, FONT_NORMAL, theme["text"], 1135, 50)
        text(surface, f"{theme['name']} mode, Auto {'ON' if self.auto_theme else 'OFF'}", FONT_SMALL, theme["muted"], 1135, 76)

    def draw_left(self, surface, theme):
        panel = pygame.Rect(10, 122, 360, 738)
        rounded(surface, panel, theme["panel"], 18, theme["line"], 1)
        text(surface, "Use SCATS site numbers or click Origin/Destination, then click a blue node on the map.", FONT_SMALL, theme["muted"], 18, 142, 330)
        text(surface, "Route Settings", FONT_H2, theme["text"], 18, 176)

        for inp in self.inputs:
            inp.draw(surface, theme)
        for dd in self.dropdowns:
            if not dd.open:
                dd.draw(surface, theme)
        self.toggle.draw(surface, theme)
        for btn in self.buttons:
            btn.draw(surface, theme)

        status_box = pygame.Rect(18, 610, 330, 96)
        rounded(surface, status_box, theme["panel2"], 12, theme["line"], 1)
        text(surface, "Status", FONT_BOLD, theme["text"], status_box.x + 14, status_box.y + 14)
        c = theme["muted"]
        if self.status_type == "success": c = theme["green"]
        elif self.status_type == "danger": c = theme["danger"]
        elif self.status_type == "warning": c = theme["warning"]
        wrap_text(surface, self.status, FONT_SMALL, c, status_box.x + 14, status_box.y + 40, status_box.width - 28)

        help_box = pygame.Rect(18, 727, 330, 110)
        rounded(surface, help_box, theme["panel2"], 12, theme["line"], 1)
        text(surface, "Map Controls", FONT_BOLD, theme["text"], help_box.x + 14, help_box.y + 14)
        wrap_text(surface,
                  "This map uses assignment SCATS nodes with OpenStreetMap road geometry. Blue nodes are dataset SCATS sites, and selected routes follow real road-network paths.",
                  FONT_SMALL, theme["muted"], help_box.x + 14, help_box.y + 40, help_box.width - 28)

    def draw_map_panel(self, surface, theme):
        panel = pygame.Rect(394, 122, 970, 540)
        rounded(surface, panel, theme["panel"], 18, theme["line"], 1)
        text(surface, "Boroondara SCATS Road Network Map", FONT_H2, theme["text"], 412, 136)
        subtitle = f"{len(self.repo.nodes)} PDF SCATS labels, {len(self.repo.data_sites)} dataset sites, {sum(len(v) for v in self.repo.graph.values())//2} route links, zoom {self.map_view.zoom:.1f}x"
        text(surface, subtitle, FONT_SMALL, theme["muted"], 412, 160)
        text(surface, "Scroll to zoom, drag to pan", FONT_SMALL, theme["muted"], 1192, 160)
        selected_path = None
        if self.routes and 0 <= self.selected_route < len(self.routes):
            selected_path = self.routes[self.selected_route]["path"]
        self.map_view.draw(surface, theme, selected_path, clean_scats(self.origin.value), clean_scats(self.dest.value))

    def draw_results(self, surface, theme):
        panel = pygame.Rect(394, 680, 970, 180)
        rounded(surface, panel, theme["panel"], 18, theme["line"], 1)
        text(surface, "Top-K Route Results", FONT_H2, theme["text"], 412, 702)
        if not self.routes:
            text(surface, "No route calculated yet. Choose origin, destination, date/time and click Calculate.", FONT_NORMAL, theme["muted"], 412, 742)
            return
        y = 724
        for i, rdata in enumerate(self.routes[:5]):
            card = pygame.Rect(412 + i * 182, y, 170, 88)
            selected = i == self.selected_route
            rounded(surface, card, theme["panel2"] if not selected else theme["input"], 12, theme["primary"] if selected else theme["line"], 2 if selected else 1)
            text(surface, f"Route {i+1}", FONT_BOLD, theme["text"], card.x + 10, card.y + 9)
            text(surface, f"{rdata['estimated_time_min']} min", FONT_NORMAL, theme["green"], card.x + 10, card.y + 31)
            text(surface, f"{rdata['distance_km']} km", FONT_SMALL, theme["muted"], card.x + 10, card.y + 52)
            text(surface, f"flow {rdata['avg_flow']} veh/hr", FONT_SMALL, theme["muted"], card.x + 10, card.y + 68)
        selected = self.routes[self.selected_route]
        text(surface, "Path: " + " → ".join(selected["path"]), FONT_SMALL, theme["muted"], 412, 822, 920)

    def draw(self, surface):
        theme = self.theme
        surface.fill(theme["bg"])
        self.draw_header(surface, theme)
        self.draw_left(surface, theme)
        self.draw_map_panel(surface, theme)
        self.draw_results(surface, theme)
        # Draw open dropdowns last so they sit above cards.
        for dd in self.dropdowns:
            if dd.open:
                dd.draw(surface, theme)

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                self.handle_event(event)
            self.draw(SCREEN)
            pygame.display.flip()
            CLOCK.tick(FPS)


if __name__ == "__main__":
    try:
        app = TBRGSGUI()
        app.run()
    except Exception as exc:
        print("TBRGS GUI failed to start:", exc)
        raise
