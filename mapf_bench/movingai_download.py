# SPDX-License-Identifier: GPL-3.0-or-later
"""Download the MovingAI MAPF benchmark (maps and official random scenarios).

    python -m mapf_bench.movingai_download                  # all maps and scenarios
    python -m mapf_bench.movingai_download --maps random-32-32-20 room-32-32-4
    python -m mapf_bench.movingai_download --map-zip mapf-map.zip --scen-zip mapf-scen-random.zip

The archives of https://movingai.com/benchmarks/mapf/ (Stern et al., SoCS
2019) are extracted into benchmarks/movingai/maps and benchmarks/movingai/scens.
With the official scenarios present, the MovingAI suites of the local runner
use them (`<map>-random-<k>.scen`) instead of generated ones. Check the URLs
below against the benchmark page if a download fails; --url-base or local
archives (--map-zip, --scen-zip) can be used instead.
"""
import argparse
import io
import os
import sys
import urllib.request
import zipfile

from .movingai import ROOT

URL_BASE = "https://movingai.com/benchmarks/mapf/"
MAP_ZIP = "mapf-map.zip"
SCEN_ZIP = "mapf-scen-random.zip"


def _fetch(src):
    if os.path.exists(src):
        return open(src, "rb").read()
    print(f"downloading {src} ...", flush=True)
    with urllib.request.urlopen(src, timeout=120) as r:
        return r.read()


def _extract(data, suffix, dest, maps=None):
    os.makedirs(dest, exist_ok=True)
    n = 0
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for info in z.infolist():
            name = os.path.basename(info.filename)
            if not name.endswith(suffix) or info.is_dir():
                continue
            stem = name[: -len(suffix)]
            mapname = stem.split("-random-")[0] if suffix == ".scen" else stem
            if maps and mapname not in maps:
                continue
            with open(os.path.join(dest, name), "wb") as fh:
                fh.write(z.read(info))
            n += 1
    return n


def download(maps=None, url_base=URL_BASE, map_zip=None, scen_zip=None, root=ROOT):
    nm = _extract(_fetch(map_zip or url_base + MAP_ZIP), ".map", os.path.join(root, "maps"), maps)
    ns = _extract(_fetch(scen_zip or url_base + SCEN_ZIP), ".scen", os.path.join(root, "scens"), maps)
    print(f"{nm} maps and {ns} scenarios in {root}")
    return nm, ns


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps", nargs="+", help="only these maps (names without .map)")
    ap.add_argument("--url-base", default=URL_BASE)
    ap.add_argument("--map-zip", help="local copy of mapf-map.zip")
    ap.add_argument("--scen-zip", help="local copy of mapf-scen-random.zip")
    a = ap.parse_args(argv)
    try:
        download(set(a.maps) if a.maps else None, a.url_base, a.map_zip, a.scen_zip)
    except Exception as e:
        print(f"download failed: {e}\nDownload the archives from {URL_BASE} by hand and pass "
              "them with --map-zip and --scen-zip.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
