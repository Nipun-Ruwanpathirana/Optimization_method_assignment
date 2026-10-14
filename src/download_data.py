"""Download the CVRPLIB instances listed in data/instances.csv.

For every instance we save two small text files in data/raw/:
    <name>.vrp  - the instance (coordinates, demands, capacity)
    <name>.sol  - the best-known solution (routes + cost), used as the reference value

Usage:
    python -m src.download_data            # download missing files only
    python -m src.download_data --force    # download everything again
"""
import argparse
import csv
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSTANCE_LIST = ROOT / "data" / "instances.csv"
RAW_DIR = ROOT / "data" / "raw"

BASE_URL = "https://galgos.inf.puc-rio.br/cvrplib/index.php/en/download"


def download(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "IT5082-assignment"})
    with urllib.request.urlopen(request, timeout=60) as response:
        target.write_bytes(response.read())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(INSTANCE_LIST, newline="") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        name, cvrplib_id = row["name"], row["cvrplib_id"]
        for kind, ext in (("instance", "vrp"), ("bks", "sol")):
            target = RAW_DIR / f"{name}.{ext}"
            if target.exists() and not args.force:
                continue
            download(f"{BASE_URL}/{kind}/{cvrplib_id}", target)
            print(f"downloaded {target.relative_to(ROOT)}")
            time.sleep(0.5)  # be polite to the CVRPLIB server

    print(f"{len(rows)} instances ready in {RAW_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
