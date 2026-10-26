"""Package completed deliverables without redistributing ACDC images or masks.

Run from the repository root after inspecting the report and presentation.
The archive includes code, measured aggregate evidence, our trained small head,
abstract predicted-mask figures, and the private classroom report/deck.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--presentation", type=Path, required=True)
    parser.add_argument("--canva-url", required=True)
    args = parser.parse_args()
    canva = urlparse(args.canva_url)
    if canva.scheme != "https" or canva.netloc != "www.canva.com" or not canva.path.startswith(("/design/", "/d/")):
        raise ValueError("Supply the verified final Canva design link")
    report = ROOT / "outputs/report/MAAI901_Technical_Report_Ayman_Lakhnati.pdf"
    presentation = args.presentation.resolve()
    for required in [report, presentation, ROOT / "working/experiments_complete.txt"]:
        if not required.is_file():
            raise FileNotFoundError(required)
    primary = json.loads((ROOT / "results/summary.json").read_text())
    dino = json.loads((ROOT / "results/dinov3/summary.json").read_text())
    train = json.loads((ROOT / "outputs/dinov3_dense/training_config.json").read_text())
    if primary["dataset"]["images"] != 406 or dino["validation_images"] != 406:
        raise ValueError("Both complete validation evaluations are required")
    if train["status"] != "completed" or not train["completed_fixed_full_protocol"]:
        raise ValueError("Completed full training protocol required")
    # Recompute evidence before packaging. A stale hand-edited summary must not
    # become a submission merely because the completion flags still exist.
    verification = subprocess.run(
        [sys.executable, str(ROOT / "scripts/verify_results.py"), "--root", str(ROOT)],
        capture_output=True, text=True,
    )
    if verification.returncode:
        raise ValueError("Evidence verification failed: " + (verification.stderr or verification.stdout).strip())
    report_receipt = json.loads((ROOT / "outputs/report/build_receipt.json").read_text())
    word_count = report_receipt["word_count_including_captions_references_tables"]
    if not isinstance(word_count, int) or not 0 < word_count <= 2500:
        raise ValueError("The report must have a verified word count at or below 2,500")
    if report_receipt.get("pdf_sha256") != hashlib.sha256(report.read_bytes()).hexdigest():
        raise ValueError("Report bytes differ from the report build receipt")

    files = set()
    for name in ["README.md", "requirements.txt", "pytest.ini", ".gitignore"]:
        files.add(ROOT / name)
    for folder in ["src", "roadseg", "scripts", "tests", "docs", "licenses", "notebooks", ".github"]:
        for candidate in (ROOT / folder).rglob("*"):
            if candidate.is_file() and "__pycache__" not in candidate.parts and candidate.suffix in {
                ".py", ".md", ".mjs", ".txt", ".ipynb", ".ps1", ".pdf", ".yml", ".yaml"
            }:
                files.add(candidate)
    for folder in ["results", "outputs/dinov3_tuning", "outputs/dinov3_dense"]:
        for candidate in (ROOT / folder).rglob("*"):
            relative = candidate.relative_to(ROOT)
            if not candidate.is_file() or any(part in {"predictions", "private", "sample_output"} for part in relative.parts):
                continue
            if candidate.suffix in {".json", ".csv", ".npz", ".xml"}:
                files.add(candidate)
            elif relative.parts[:2] == ("results", "figures") and candidate.suffix in {".png", ".pdf"}:
                files.add(candidate)
    files.add(ROOT / "outputs/dinov3_dense/head.pt")
    files.add(report)
    files.add(ROOT / "outputs/report/report.md")
    files.update(p for p in (ROOT / "build/report/monochrome").glob("*") if p.is_file() and p.suffix in {".png", ".pdf", ".json"})
    files.add(presentation)
    files.update(p for p in (ROOT / "assets/presentation").rglob("*") if p.is_file() and p.suffix in {".png", ".md", ".json"})
    for name in ["speaker_notes.md"]:
        source = ROOT / "build/presentation" / name
        if source.is_file():
            files.add(source)

    # Only explicit allowlisted outputs enter the archive. Never walk data/,
    # checkpoints/, cache/, working/, or raw prediction directories.
    records = []
    output = ROOT / "submission/MAAI901_Ayman_Lakhnati_VRU_Segmentation.zip"
    output.parent.mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".zip", delete=False) as staged:
        staged_path = Path(staged.name)
    with ZipFile(staged_path, "w", ZIP_DEFLATED, compresslevel=6) as archive:
        for source in sorted(files):
            source = source.resolve()
            relative = source.relative_to(ROOT).as_posix()
            if relative.startswith(("data/", "checkpoints/", "cache/", "working/")):
                raise ValueError(f"Restricted artifact in archive: {relative}")
            payload = source.read_bytes()
            archive.writestr(relative, payload)
            records.append({"path": relative, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
        handoff = (
            "MAAI901: Transformer Segmentation of Vulnerable Road Users in Adverse Conditions\n\n"
            "Technical report: outputs/report/MAAI901_Technical_Report_Ayman_Lakhnati.pdf\n"
            f"Editable Canva presentation: {args.canva_url}\n"
            f"PowerPoint copy: {presentation.relative_to(ROOT).as_posix()}\n"
            "GitHub: https://github.com/AymanLakhnati/Adverse-Condition-VRU-Segmentation\n"
            "Code and reproducibility: README.md and notebooks/Adverse_Condition_VRU_Segmentation.ipynb\n"
            "Measured results: results/summary.json and results/dinov3/summary.json\n"
            "Visualizations and abstract sample predictions: results/figures/\n"
            "Presentation preparation: docs/PRESENTATION_DEFENSE.md\n\n"
            "Grading evidence map: docs/GRADING_EVIDENCE.md\n"
            "Verify saved results without data/GPU: python scripts/verify_results.py\n"
            "Folder prediction and output safeguards: docs/SOFTWARE_GUIDE.md\n\n"
            "ACDC RGB images, ground-truth masks and reconstructive overlays are excluded under its license.\n"
            "Download data and upstream model weights from official sources using the documented commands.\n"
        )
        archive.writestr("START_HERE.txt", handoff)
        archive.writestr("SUBMISSION_MANIFEST.json", json.dumps({"files": records, "canva_url": args.canva_url}, indent=2))
    with ZipFile(staged_path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Archive integrity check failed")
        for record in records:
            payload = archive.read(record["path"])
            if len(payload) != record["bytes"] or hashlib.sha256(payload).hexdigest() != record["sha256"]:
                raise RuntimeError(f"Archive member verification failed: {record['path']}")
    os.replace(staged_path, output)
    print(json.dumps({"archive": str(output), "files": len(records), "bytes": output.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
