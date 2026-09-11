#!/usr/bin/env python
"""Smoke test for the Flask dashboard (``web/app.py``) and the interactive 3D viewer.

Runs the real pipeline, checks that every expected key is present, that the
three HTTP routes plus ``/api/view3d`` behave, and that the viewer builds its
four scenes.  Nothing is spawned here: the ``/api/view3d`` checks only exercise
the error paths, so no Open3D window opens on your desktop.  Intended to be run
from the project root::

    python scripts/check_dashboard.py

Exits with status 0 on success, 1 on the first failed check.
"""
from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg  # noqa: E402

import web.app as dashboard  # noqa: E402  (path set above)
from web import view3d  # noqa: E402


def check(condition: bool, message: str) -> None:
    status = "ok  " if condition else "FAIL"
    print(f"[{status}] {message}")
    if not condition:
        raise SystemExit(1)


def main() -> int:
    print("running pipeline ...")
    payload = dashboard.run_pipeline(force=True)

    images = payload["images"]
    metrics = payload["metrics"]
    memory = payload["memory"]

    # 2D matplotlib panels always render; the four Open3D PNG panels only exist
    # when offscreen OpenGL worked (render3d_error is then a non-empty reason).
    expected = ["adaptive", "uniform", "drivability", "density2d", "timing", "memory",
                "comparison"]
    if not payload["render3d_error"]:
        expected += list(dashboard.VIEWS)
    else:
        print(f"[note] offscreen 3D rendering unavailable: {payload['render3d_error']}")
    check(sorted(images) == sorted(expected),
          f"{len(expected)} PNG panels rendered: {sorted(images)}")
    for name, b64 in images.items():
        check(isinstance(b64, str) and len(b64) > 1000,
              f"panel '{name}' is a non-trivial base64 PNG ({len(b64)} chars)")
        raw = base64.b64decode(b64)
        check(raw[:8] == b"\x89PNG\r\n\x1a\n",
              f"panel '{name}' starts with the PNG signature")
        img = mpimg.imread(io.BytesIO(raw))
        check(img.ndim == 3 and img.shape[0] > 200 and img.shape[1] > 200,
              f"panel '{name}' decodes to a {img.shape[1]}x{img.shape[0]} image")

    check(metrics["points_raw"] > 0, f"raw points = {metrics['points_raw']:,}")
    check(metrics["points_projected"] >= metrics["points_filtered"] * 0.9,
          f"projected {metrics['points_projected']:,} of "
          f"{metrics['points_filtered']:,} filtered points")
    check(metrics["cells_occupied"] > 0,
          f"occupied adaptive cells = {metrics['cells_occupied']:,}")
    check(metrics["cells_uniform"] > 0,
          f"occupied uniform cells = {metrics['cells_uniform']:,}")
    check(metrics["total_ms"] > 0, f"end-to-end latency = {metrics['total_ms']:.1f} ms")
    check(metrics["foveation_ratio"] > 1.0,
          f"foveation ratio = {metrics['foveation_ratio']:.1f}x "
          f"(cells {metrics['cell_sizes']})")

    check(memory["reduction_vs_uniform25d_pct"] > 0,
          f"smaller than dense uniform 2.5D by "
          f"{memory['reduction_vs_uniform25d_pct']:.1f}%")
    check(memory["reduction_vs_uniform3d_pct"] > 0,
          f"smaller than dense uniform 3D voxels by "
          f"{memory['reduction_vs_uniform3d_pct']:.1f}%")

    check(len(payload["bands"]) == metrics["bands"],
          f"{len(payload['bands'])} band rows")
    check(len(payload["classes"]) == metrics["classes"] + 1,
          f"{len(payload['classes'])} semantic rows (incl. unknown)")
    check(len(payload["tables"]["latency"]) > 0 and len(payload["tables"]["memory"]) > 0,
          "latency + memory tables populated")

    # --- discrete objects (grouped non-ground cells) ------------------------ #
    objects = payload["objects"]
    check(objects["enabled"] is True, "object extraction is enabled in the payload")
    check(isinstance(objects["items"], list) and isinstance(objects["rows"], list),
          "object payload carries both raw items and formatted rows")
    check(len(objects["rows"]) == len(objects["items"]) == objects["count"],
          f"{objects['count']} object blobs grouped from "
          f"{objects['cells_considered']:,} candidate cells")
    check(objects["count"] >= 1, "at least one object blob found on this frame")
    for item in objects["items"]:
        check(item["class_name"] and item["range_m"] >= 0.0 and item["distance_m"] >= item["range_m"],
              f"object {item['id']} '{item['class_name']}' range {item['range_m']:.1f} m "
              f"<= distance {item['distance_m']:.1f} m")
        check(item["cells"] >= objects["min_cells"],
              f"object {item['id']} covers {item['cells']} cells (>= min_cells)")
        check(item["height_m"] >= 0.0 and 0.0 <= item["confidence"] <= 1.0,
              f"object {item['id']} height {item['height_m']:.2f} m, "
              f"confidence {item['confidence']:.2f}")
    distances = [item["distance_m"] for item in objects["items"]]
    check(distances == sorted(distances), "object rows are ordered nearest-first")
    check([item["id"] for item in objects["items"]] == list(range(1, objects["count"] + 1)),
          "object ids are 1..N in distance order")

    client = dashboard.app.test_client()
    for route in ("/", "/api/data", "/healthz"):
        response = client.get(route)
        check(response.status_code == 200, f"GET {route} -> 200 ({len(response.data):,} bytes)")

    # --- interactive Open3D viewer (option: dashboard button -> native window) ---
    rules = {rule.rule for rule in dashboard.app.url_map.iter_rules()}
    check("/api/view3d" in rules, "the /api/view3d route is registered")
    check(payload["views"] == list(dashboard.VIEWS),
          f"payload advertises the viewer views: {payload['views']}")
    check(len(payload["viewer"]["controls"]) >= 5,
          f"{len(payload['viewer']['controls'])} documented window controls")

    clouds = view3d.build_scene()          # the exact path the button spawns
    check(sorted(clouds) == sorted(dashboard.VIEWS),
          f"viewer builds all four views: {sorted(clouds)}")
    for name in dashboard.VIEWS:
        pts, cols = clouds[name]
        check(pts.shape[0] > 0 and pts.shape[1] == 3 and cols.shape == pts.shape,
              f"view '{name}': {pts.shape[0]:,} points, colours {cols.shape}")
        check(float(cols.min()) >= 0.0 and float(cols.max()) <= 1.0,
              f"view '{name}' colours are inside [0, 1]")

    # error paths must be refused *without* spawning a window
    missing = view3d.launch_viewer("adaptive_semantic",
                                   dataset=ROOT / "data" / "does_not_exist.bin")
    check(missing["ok"] is False and "not found" in missing["error"],
          f"launch_viewer refuses a missing dataset: {missing['error']}")
    unknown = view3d.launch_viewer("not_a_view")
    check(unknown["ok"] is False and unknown["views"] == list(dashboard.VIEWS),
          f"launch_viewer refuses an unknown view: {unknown['error']}")
    bad = client.get("/api/view3d?view=not_a_view")
    check(bad.status_code == 400, f"GET /api/view3d?view=not_a_view -> 400")
    check(bad.get_json()["error"].startswith("unknown view"),
          f"the 400 body explains the failure: {bad.get_json()['error']}")

    # the dashboard button POSTs; a 405 here would break the browser button
    posted = client.post("/api/view3d?view=not_a_view")
    check(posted.status_code == 400,
          f"POST /api/view3d is routed (no 405): {posted.status_code}")

    html = client.get("/").get_data(as_text=True)
    check("data:image/png;base64," in html, "dashboard HTML embeds the PNG panels")
    check("Drivability layer - 3 walkable-ground tiers" in html,
          "dashboard HTML has the drivability panel")
    check("Sampling density - the foveation made visible" in html,
          "dashboard HTML has the density panel")
    check("Latency breakdown" in html and "Memory footprint comparison" in html,
          "dashboard HTML has the latency + memory panels")
    check("Adaptive bands (foveation schedule)" in html,
          "dashboard HTML has the foveation-bands panel")
    check("Discrete objects - non-ground cells grouped into blobs" in html,
          "dashboard HTML has the discrete-objects panel")
    check("measured head to head" in html and "Adaptive vs uniform" in html,
          "dashboard HTML has the adaptive-vs-uniform comparison panel")
    check(payload["comparison"]["rows"] and
          payload["comparison"]["measured"] is not None,
          f"comparison payload has {len(payload['comparison']['rows'])} measured rows")

    # Panels the template actually embeds.  The standalone adaptive/uniform 2.5D
    # map panels were folded into the 3D previews, so those two PNGs are still
    # rendered by the pipeline (and served on /api/data) but are intentionally
    # not shown on the page.
    unshown = {"adaptive", "uniform"}
    shown = [name for name in images if name not in unshown]
    check(html.count("data:image/png;base64,") == len(shown),
          f"dashboard HTML embeds all {len(shown)} shown panels "
          f"(found {html.count('data:image/png;base64,')})")
    for name in sorted(unshown):
        check(name in images,
              f"panel '{name}' is still rendered for the JSON payload but not shown")
    for name in images:
        check(f'data:image/png;base64,{{{{ p.images.{name} }}}}' not in html,
              f"panel '{name}' is interpolated, not left as a template literal")
    if not payload["render3d_error"]:
        check(html.count("3D preview - ") == len(dashboard.VIEWS),
              f"dashboard HTML shows all {len(dashboard.VIEWS)} static 3D preview panels")
        for title in payload["view_titles"].values():
            check(f"3D preview - {title}" in html,
                  f"dashboard HTML has the '{title}' 3D preview panel")

    check("Open interactive 3D" in html and "/api/view3d" in html,
          "dashboard HTML has the interactive 3D button + endpoint")
    check(all(f'data-view3d="{name}"' in html for name in dashboard.VIEWS),
          "dashboard HTML has one 3D button per view")
    check("{{" not in html and "{%" not in html,
          "dashboard HTML contains no unrendered Jinja markers")

    cached = dashboard.run_pipeline()
    check(cached is payload, "second call is served from the cache")

    print("\n--- latency ---")
    for row in payload["tables"]["latency"]:
        print(f"  {row['label']:<52} {row['value']:>10}")
    print("\n--- memory ---")
    for row in payload["tables"]["memory"]:
        print(f"  {row['label']:<52} {row['value']:>10}  ({row['note']})")
    print("\n--- bands ---")
    for b in payload["bands"]:
        print(f"  band {b['band']}: {b['rmin']:>5.1f}-{b['rmax']:<5.1f} m  "
              f"cell {b['cell'] * 100:>5.1f} cm  cells {b['cells']:>8,}  "
              f"occupied {b['occupied']:>7,}  fill {b['occupancy_pct']:>5.1f}%  "
              f"conf {b['mean_confidence']:.2f}")
    print("\n--- semantic mix ---")
    for c in payload["classes"]:
        print(f"  {c['name']:<18} {c['count']:>8,} points  {c['pct']:>5.1f}%")

    print("\n--- adaptive vs uniform head-to-head ---")
    for row in payload["comparison"]["rows"]:
        print(f"  {row['label']:<34} adaptive {row['adaptive_text']:>12}  "
              f"uniform {row['uniform_text']:>12}  {row['advantage']:>6.2f}x  "
              f"[{row['winner']}]")

    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
