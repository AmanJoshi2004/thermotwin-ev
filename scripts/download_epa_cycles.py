"""Download the EPA UDDS, HWFET and US06 drive cycles into data/cycles/.

If a URL has moved, open https://www.epa.gov/vehicle-and-fuel-emissions-testing/dynamometer-drive-schedules
and save the text files manually into data/cycles/ with the names below.
"""

from pathlib import Path
from urllib.request import urlopen

from thermotwin.bench.cycles import EPA_BASE, EPA_FILES, parse_epa

out = Path("data/cycles")
out.mkdir(parents=True, exist_ok=True)
for name, fname in EPA_FILES.items():
    url = f"{EPA_BASE}/{fname}"
    try:
        (out / fname).write_bytes(urlopen(url, timeout=30).read())
        cyc = parse_epa(out / fname)
        print(f"{name}: {len(cyc)} points, max {cyc.speed_mps.max() * 3.6:.0f} km/h -> {out / fname}")
    except Exception as e:  # noqa: BLE001
        print(f"{name}: failed ({e}). Download {url} manually into {out}/")
