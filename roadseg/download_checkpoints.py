"""Download the authors' public Cityscapes-to-ACDC HRDA and MIC archives."""
import hashlib
import json
import tarfile
from pathlib import Path


def main():
    import truststore
    truststore.inject_into_ssl()
    import gdown
    root = Path("checkpoints")
    root.mkdir(exist_ok=True)
    records = {}
    for name, file_id, folder in [
        ("hrda", "1Pwb1NAKx-WeVdhKEfafn29_JP4iEyKQM", "csHR2acdcHR_hrda_d3a68"),
        ("mic", "10RNOAyUY5nYKzIIbNTie458r9etzfvtc", "csHR2acdcHR_mic_hrda_b52f1"),
    ]:
        archive = root / f"{name}_acdc.tar.gz"
        checkpoint = root / name / folder / "iter_40000_relevant.pth"
        if not checkpoint.exists():
            if not archive.exists():
                gdown.download(id=file_id, output=str(archive), quiet=False)
            with tarfile.open(archive) as tar:
                tar.extractall(root / name, filter="data")
        h = hashlib.sha256()
        with open(checkpoint, "rb") as f:
            for block in iter(lambda: f.read(1048576), b""):
                h.update(block)
        records[name] = {"source": f"https://drive.google.com/file/d/{file_id}/view",
                         "path": str(checkpoint), "sha256": h.hexdigest(),
                         "original_code": "https://github.com/lhoyer/MIC" if name == "mic" else "https://github.com/lhoyer/HRDA"}
        print(name, checkpoint)
    (root / "provenance.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
