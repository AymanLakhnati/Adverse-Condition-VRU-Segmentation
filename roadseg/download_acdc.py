"""Stream official ACDC ZIP and retain adverse validation images only.

The official package endpoint does not support HTTP ranges. This reader uses
ZIP local headers to skip unwanted compressed members without materializing
the 15.6 GB archive. It verifies CRC32 of each retained file.
"""
import argparse
import hashlib
import json
import logging
import struct
import time
import zlib
import zipfile
from pathlib import Path, PurePosixPath

import requests

LOG = logging.getLogger("acdc-download")
BASE = "https://acdc.vision.ee.ethz.ch/api/"
PACKAGE = "6436f2259880d97633275dfc"


def ensure_labels(root):
    """Download official 127 MB label archive and verify its official MD5."""
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "gt_trainval.zip"
    expected_md5 = "54cf06e3f2d8a8c5d297dc55aa107cee"
    if not path.exists():
        response = requests.get(BASE + "getPackageUri/6436eeae9880d97633275dc9", timeout=60)
        response.raise_for_status()
        token = response.json()["token"]
        with requests.get(BASE + "downloadPackage/" + token + "/gt_trainval.zip", stream=True, timeout=(60, 180)) as r:
            r.raise_for_status()
            with open(path.with_suffix(".part"), "wb") as f:
                for chunk in r.iter_content(1048576):
                    f.write(chunk)
        path.with_suffix(".part").replace(path)
    md5 = hashlib.md5(path.read_bytes()).hexdigest()
    if md5 != expected_md5:
        raise ValueError("Official label archive MD5 mismatch")
    with zipfile.ZipFile(path) as archive:
        selected = [n for n in archive.namelist() if
                    n.endswith("_gt_labelTrainIds.png") or n == "License.pdf"]
        for name in selected:
            target = root.joinpath(*PurePosixPath(name).parts)
            if not target.exists():
                archive.extract(name, root)
    LOG.info("Official label archive MD5 verified; 2006 train/validation train-ID masks available")


class Reader:
    def __init__(self, raw):
        self.raw = raw
        self.total = 0

    def read(self, n):
        data = self.raw.read(n)
        self.total += len(data)
        if len(data) != n:
            raise EOFError(f"Truncated archive at {self.total} bytes")
        return data

    def skip(self, n):
        while n:
            size = min(n, 1024 * 1024)
            self.read(size)
            n -= size


def download(root, include_train=False):
    # Use OS trust roots on Windows without disabling TLS verification.
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    response = session.get(BASE + "getPackageUri/" + PACKAGE, timeout=60)
    response.raise_for_status()
    token = response.json()["token"]
    url = BASE + "downloadPackage/" + token + "/rgb_anon_trainvaltest.zip"
    expected = 2006 if include_train else 406
    started, kept, members = time.monotonic(), 0, 0
    with session.get(url, stream=True, timeout=(60, 180)) as response:
        response.raise_for_status()
        reader = Reader(response.raw)
        while kept < expected:
            signature = reader.read(4)
            if signature == b"PK\x01\x02":
                break
            if signature != b"PK\x03\x04":
                raise ValueError(f"Unexpected ZIP signature at {reader.total - 4}")
            fields = struct.unpack("<5H3I2H", reader.read(26))
            version, flags, method, _, _, crc, compressed, uncompressed, name_len, extra_len = fields
            name = reader.read(name_len).decode("utf-8" if flags & 0x800 else "cp437")
            extra = reader.read(extra_len)
            if flags & 8:
                raise ValueError("ZIP uses streaming data descriptors; local sizes unavailable")
            if compressed == 0xFFFFFFFF or uncompressed == 0xFFFFFFFF:
                offset = 0
                while offset + 4 <= len(extra):
                    kind, size = struct.unpack_from("<HH", extra, offset)
                    value = extra[offset + 4:offset + 4 + size]
                    offset += 4 + size
                    if kind == 1:
                        values = iter(struct.unpack("<" + "Q" * (len(value) // 8), value))
                        if uncompressed == 0xFFFFFFFF:
                            uncompressed = next(values)
                        if compressed == 0xFFFFFFFF:
                            compressed = next(values)
                        break
            parts = PurePosixPath(name).parts
            retain = (len(parts) >= 5 and parts[0] == "rgb_anon" and
                      parts[1] in ("fog", "night", "rain", "snow") and
                      parts[2] in (("train", "val") if include_train else ("val",)) and
                      name.endswith("_rgb_anon.png"))
            if retain:
                target = root.joinpath(*parts)
                existing = target.read_bytes() if target.exists() else None
                if existing is not None and len(existing) == uncompressed and zlib.crc32(existing) & 0xFFFFFFFF == crc:
                    reader.skip(compressed)
                else:
                    content = reader.read(compressed)
                    raw = content if method == 0 else zlib.decompress(content, -15) if method == 8 else None
                    if raw is None or len(raw) != uncompressed or zlib.crc32(raw) & 0xFFFFFFFF != crc:
                        raise ValueError(f"Corrupt member: {name}")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(raw)
                kept += 1
                LOG.info("Retained %d/%d %s (stream %.2f GB, %.1f min)", kept, expected, name,
                         reader.total / 1e9, (time.monotonic() - started) / 60)
            else:
                reader.skip(compressed)
            members += 1
            if members % 100 == 0:
                LOG.info("Read %d members, retained %d, streamed %.2f GB, latest %s", members, kept, reader.total / 1e9, name)
    if kept != expected:
        raise ValueError(f"Expected {expected} validation images, extracted {kept}")
    (root / "download_provenance.json").write_text(json.dumps({
        "source": "https://acdc.vision.ee.ethz.ch/download", "package_id": PACKAGE,
        "package_name": "rgb_anon_trainvaltest.zip", "retained_images": kept,
        "include_training_images": include_train,
        "bytes_streamed": reader.total, "retained_crc32_verified": True,
        "full_archive_md5_verified": False, "elapsed_seconds": time.monotonic() - started,
        "terms": "Official ACDC research license applies; raw data excluded from repository"
    }, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("data/acdc"))
    p.add_argument("--include-train", action="store_true")
    p.add_argument("--labels-only", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ensure_labels(args.root)
    if not args.labels_only:
        for attempt in range(1, 6):
            try:
                download(args.root, args.include_train)
                break
            except (requests.RequestException, EOFError, OSError) as error:
                LOG.warning("Download attempt %d failed: %s", attempt, type(error).__name__)
                if attempt == 5:
                    raise
                time.sleep(15)
