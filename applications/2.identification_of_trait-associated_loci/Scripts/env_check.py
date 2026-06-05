#!/usr/bin/env python3
"""Validate local data files, model files, and Python dependencies."""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import shutil
import tempfile
import urllib.request
from pathlib import Path


def project_root(config_path: Path) -> Path:
    return config_path.resolve().parent


def rel(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def download_reference(url: str, fasta_path: Path) -> None:
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_download = fasta_path.with_suffix(fasta_path.suffix + ".download")
    print(f"[INFO] downloading reference FASTA: {url}")
    urllib.request.urlretrieve(url, tmp_download)

    try:
        bgzip_reference(tmp_download, fasta_path)
    except Exception as exc:
        tmp_download.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to convert downloaded FASTA to BGZF: {exc}") from exc


def bgzip_reference(source_gzip: Path, target_bgzip: Path) -> None:
    import pysam

    tmp_bgzip = target_bgzip.with_suffix(target_bgzip.suffix + ".bgz")
    with tempfile.NamedTemporaryFile(suffix=".fa", delete=False) as tmp_plain:
        tmp_plain_path = Path(tmp_plain.name)
    try:
        with gzip.open(source_gzip, "rb") as src, tmp_plain_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        pysam.tabix_compress(str(tmp_plain_path), str(tmp_bgzip), force=True)
        tmp_bgzip.replace(target_bgzip)
    finally:
        tmp_plain_path.unlink(missing_ok=True)
        if source_gzip != target_bgzip:
            source_gzip.unlink(missing_ok=True)


def ensure_reference_indexes(fasta_path: Path) -> None:
    try:
        import pysam

        print(f"[INFO] building FASTA indexes for: {fasta_path}")
        pysam.faidx(str(fasta_path))
    except Exception as exc:
        print(f"[WARN] FASTA indexing failed, attempting BGZF conversion: {exc}")
        try:
            backup = fasta_path.with_suffix(fasta_path.suffix + ".plain-gzip")
            fasta_path.replace(backup)
            bgzip_reference(backup, fasta_path)
            pysam.faidx(str(fasta_path))
        except Exception as convert_exc:
            raise RuntimeError(
                "Failed to build FASTA indexes. The reference FASTA must be readable as gzip "
                "and convertible to BGZF."
            ) from convert_exc


def validate_reference_fasta(fasta_path: Path) -> bool | None:
    try:
        import pysam
    except ImportError:
        print("[WARN] reference FASTA random-access check skipped because pysam is not installed")
        return None

    try:
        fasta = pysam.FastaFile(str(fasta_path))
        if not fasta.references:
            fasta.close()
            raise RuntimeError("no contigs found")
        fasta.close()
        print("[OK] reference FASTA is readable with random access")
        return True
    except Exception as exc:
        print(f"[MISS] reference FASTA random-access check failed: {exc}")
        print("       Expected a BGZF-compressed FASTA with .fai and .gzi indexes.")
        return False


def check_reference_contigs(root: Path, cfg: dict) -> list[str] | None:
    missing = []
    try:
        import pysam

        fasta_path = rel(root, cfg["paths"]["reference_fasta"])
        regions_path = rel(root, cfg["paths"]["candidate_regions"])
        fasta = pysam.FastaFile(str(fasta_path))
        references = set(fasta.references)
        for line in regions_path.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            chrom = line.split()[0].replace("chr", "").replace("Chr", "")
            candidates = {chrom, f"chr{chrom}", f"Chr{chrom}"}
            if references.isdisjoint(candidates):
                missing.append(chrom)
        fasta.close()
    except Exception as exc:
        print(f"[WARN] reference contig check failed: {exc}")
        return None
    return sorted(set(missing))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--download-reference",
        action="store_true",
        help="Download the reference FASTA configured in the config file when missing and build .fai/.gzi indexes.",
    )
    parser.add_argument(
        "--repair-reference-index",
        action="store_true",
        help="Build missing reference FASTA indexes for an existing BGZF-compressed FASTA.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    root = project_root(config_path)
    cfg = json.loads(config_path.read_text(encoding="utf-8"))

    print(f"Project root: {root}")

    fasta_path = rel(root, cfg["paths"]["reference_fasta"])
    fai_path = rel(root, cfg["paths"]["reference_fasta_fai"])
    gzi_path = rel(root, cfg["paths"]["reference_fasta_gzi"])
    reference_url = cfg.get("workflow", {}).get("reference_download", {}).get("url", "")
    if not fasta_path.exists():
        print(f"[MISS] reference FASTA: {fasta_path}")
        if reference_url:
            print(f"[INFO] download URL: {reference_url}")
            print("[INFO] run `bash 0.env_check.sh --download-reference` to download and index it automatically.")
    if args.download_reference and not fasta_path.exists():
        if not reference_url:
            print("[WARN] --download-reference was requested, but no reference URL is configured.")
        else:
            download_reference(reference_url, fasta_path)
    if fasta_path.exists() and (args.repair_reference_index or not fai_path.exists() or not gzi_path.exists()):
        ensure_reference_indexes(fasta_path)

    required_paths = [
        "vcf",
        "vcf_index",
        "phenotype",
        "reference_fasta",
        "reference_fasta_fai",
        "reference_fasta_gzi",
        "gene_annotation",
        "grain_length_gene_annotation",
        "grain_length_vcf",
        "grain_length_vcf_index",
        "grain_length_phenotype",
        "grain_length_regions",
        "candidate_regions",
        "model",
    ]
    missing = []
    for key in required_paths:
        path = rel(root, cfg["paths"][key])
        ok = path.exists()
        print(f"[{'OK' if ok else 'MISS'}] {key}: {path}")
        if not ok:
            missing.append(str(path))

    if fasta_path.exists() and fai_path.exists() and gzi_path.exists():
        reference_ok = validate_reference_fasta(fasta_path)
        if reference_ok is False:
            missing.append(f"reference FASTA random access: {fasta_path}")
        missing_contigs = check_reference_contigs(root, cfg)
        if missing_contigs is None:
            pass
        elif missing_contigs:
            missing.append(f"reference contigs for candidate regions: {', '.join(missing_contigs)}")
            print(f"[MISS] reference contigs for candidate regions: {', '.join(missing_contigs)}")
        else:
            print("[OK] reference contigs cover candidate regions")

    model_dir = rel(root, cfg["paths"]["model"])
    for filename in ("config.json", "tokenizer.json", "model.safetensors"):
        path = model_dir / filename
        ok = path.exists()
        print(f"[{'OK' if ok else 'MISS'}] model/{filename}")
        if not ok:
            missing.append(str(path))

    module_missing = []
    for module in cfg["environment"].get("required_python_modules", []):
        ok = importlib.util.find_spec(module) is not None
        print(f"[{'OK' if ok else 'MISS'}] python module: {module}")
        if not ok:
            module_missing.append(module)

    try:
        import torch

        print(f"[INFO] torch: {torch.__version__}")
        print(f"[INFO] cuda available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[INFO] cuda device count: {torch.cuda.device_count()}")
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"[WARN] torch CUDA diagnostic failed: {exc}")

    if missing or module_missing:
        print("\nEnvironment check failed.")
        if missing:
            print("Missing files/directories:")
            for item in missing:
                print(f"  - {item}")
        if module_missing:
            print("Missing Python modules:")
            for item in module_missing:
                print(f"  - {item}")
        return 1

    print("\nEnvironment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
