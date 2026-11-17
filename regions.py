"""
regions.py - Ilgi alanlari (AOI): bakis noktasini ekrandaki bolgelere atar, isabet oranini olcer.

Bolge kaynaklari:
  grid:3x3         ekrani esit hucrelere boler
  aoi.json         elle tanimli bolgeler: [{"name": "reklam", "x": 100, "y": 200, "w": 400, "h": 300}, ...]
  auto             ekran goruntusunden metin bloklari / resimler / dugmeler otomatik bulunur
"""
import json
import threading
import time

import cv2
import numpy as np


class Region:
    def __init__(self, name, x, y, w, h):
        self.name, self.x, self.y, self.w, self.h = name, float(x), float(y), float(w), float(h)

    def contains(self, px, py):
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h

    def distance(self, px, py):
        dx = max(self.x - px, 0, px - (self.x + self.w))
        dy = max(self.y - py, 0, py - (self.y + self.h))
        return (dx * dx + dy * dy) ** 0.5

    def __repr__(self):
        return f"Region({self.name!r}, {self.x:.0f}, {self.y:.0f}, {self.w:.0f}, {self.h:.0f})"


def grid_regions(cols, rows, W, H):
    return [Region(f"r{r + 1}c{c + 1}", c * W / cols, r * H / rows, W / cols, H / rows)
            for r in range(rows) for c in range(cols)]


def load_regions(path):
    with open(path, encoding="utf-8") as f:
        return [Region(d.get("name", f"aoi{i + 1}"), d["x"], d["y"], d["w"], d["h"]) for i, d in enumerate(json.load(f))]


def parse_spec(spec, W, H):
    """'grid:3x3', 'auto' ya da JSON dosya yolu -> (bolgeler, AutoRegions | None)."""
    if spec.startswith("grid:"):
        cols, rows = (int(v) for v in spec[5:].lower().split("x"))
        return grid_regions(cols, rows, W, H), None
    if spec == "auto":
        auto = AutoRegions()
        auto.start()
        return [], auto
    return load_regions(spec), None


def nearest(regions, px, py, snap):
    """Noktayi iceren bolge (ic ice ise en kucugu); yoksa `snap` px icindeki en yakin bolge."""
    inside = [r for r in regions if r.contains(px, py)]
    if inside:
        return min(inside, key=lambda r: r.w * r.h)
    best = min(regions, key=lambda r: r.distance(px, py), default=None)
    return best if best is not None and best.distance(px, py) <= snap else None


class RegionAssigner:
    """Bolge degisimini ancak `hold` kare ust uste yeni bolgede kalinca kabul eder (titremeyi bastirir)."""

    def __init__(self, snap=80, hold=4):
        self.snap, self.hold = snap, hold
        self.current = self.pending = None
        self.pending_name = None
        self.count = 0

    def __call__(self, regions, px, py):
        r = nearest(regions, px, py, self.snap)
        name = r.name if r else None
        if name == (self.current.name if self.current else None):
            self.count = 0
        elif self.count and name == self.pending_name:
            self.count += 1
        else:
            self.pending, self.pending_name, self.count = r, name, 1
        if self.count >= self.hold:
            self.current, self.count = self.pending, 0
        return self.current


def hit_rate(P, T, regions, snap=0):
    """Hedefi bir bolgenin icinde olan orneklerden, tahmini ayni bolgeye dusenlerin orani."""
    hits = total = 0
    for (px, py), (tx, ty) in zip(P, T):
        target = nearest(regions, tx, ty, 0)
        if target is None:
            continue
        total += 1
        got = nearest(regions, px, py, snap)
        hits += got is not None and got.name == target.name
    return hits / total if total else float("nan")


def detect_screen_regions(img, min_area=5000):
    """Ekran goruntusunden icerik bloklarini bul: kelimeler satira, satirlar paragrafa birlesir."""
    H, W = img.shape[:2]
    small = cv2.resize(img, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    blocks = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 7)))
    contours, _ = cv2.findContours(blocks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = (v * 2 for v in cv2.boundingRect(c))
        if w * h >= min_area and w * h < 0.5 * W * H:
            boxes.append([x, y, x + w, y + h])
    boxes = _merge_overlapping(boxes)
    boxes.sort(key=lambda b: (b[1] // 40, b[0]))  # okuma sirasi
    return [Region(f"oge{i + 1}", b[0], b[1], b[2] - b[0], b[3] - b[1]) for i, b in enumerate(boxes)]


def _merge_overlapping(boxes):
    merged = True
    while merged:
        merged = False
        out = []
        for b in boxes:
            for o in out:
                if b[0] < o[2] and o[0] < b[2] and b[1] < o[3] and o[1] < b[3]:
                    o[:] = [min(o[0], b[0]), min(o[1], b[1]), max(o[2], b[2]), max(o[3], b[3])]
                    merged = True
                    break
            else:
                out.append(list(b))
        boxes = out
    return boxes


class AutoRegions(threading.Thread):
    """Ekrani `period` saniyede bir yakalayip bolgeleri gunceller."""

    def __init__(self, period=2.0):
        super().__init__(daemon=True)
        self.period = period
        self.regions = []
        self.running = True

    def run(self):
        import mss
        with getattr(mss, "MSS", mss.mss)() as sct:
            mon = sct.monitors[1]
            while self.running:
                img = np.asarray(sct.grab(mon))[:, :, :3]
                self.regions = detect_screen_regions(np.ascontiguousarray(img))
                time.sleep(self.period)
