"""
Audit du volume RunPod pour le pipeline Medusa I2I.

Scanne le network volume et identifie les fichiers non utilises
par le pipeline (modeles obsoletes, caches orphelins, outputs anciens).

Usage:
    python scripts/audit_volume.py [--volume /runpod-volume] [--json] [--max-age-days 30]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[audit] %(message)s")
log = logging.getLogger("audit")

# --- Fichiers attendus par le pipeline ---

# Sous-dossiers du repo Qwen attendus (hors transformer/*.safetensors)
QWEN_EXPECTED_SUBDIRS = {
    "text_encoder",
    "text_encoder_2",
    "tokenizer",
    "tokenizer_2",
    "vae",
    "scheduler",
    "transformer",  # dossier config seulement, pas de .safetensors
}

QWEN_EXPECTED_ROOT_FILES = {
    "model_index.json",
    "config.json",
    "README.md",
}

FP8_FILENAMES = {
    "qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_4steps_v1.0.safetensors",
    "qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_8steps_v1.0.safetensors",
}

# Anciens fichiers FP8 (renommes upstream)
OBSOLETE_FP8_FILENAMES = {
    "qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning.safetensors",
}

# Dossiers fonctionnels sur le volume
FUNCTIONAL_DIRS = {"models", "output", "cache"}


def format_size(size_bytes: int) -> str:
    """Formate une taille en bytes vers une representation lisible."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def dir_size(path: Path) -> int:
    """Taille totale d'un dossier en bytes."""
    total = 0
    try:
        for entry in path.rglob("*"):
            if entry.is_file():
                try:
                    total += entry.stat().st_size
                except OSError:
                    pass
    except PermissionError:
        pass
    return total


def file_age_days(path: Path) -> float:
    """Age du fichier en jours depuis derniere modification."""
    try:
        mtime = path.stat().st_mtime
        return (time.time() - mtime) / 86400
    except OSError:
        return 0.0


@dataclass
class AuditEntry:
    """Un element identifie par l'audit."""

    path: str
    size_bytes: int
    category: str  # "obsolete", "orphan", "stale_output", "stale_cache", "unknown"
    reason: str
    age_days: float = 0.0

    @property
    def size_human(self) -> str:
        return format_size(self.size_bytes)

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "size_human": self.size_human,
            "category": self.category,
            "reason": self.reason,
            "age_days": round(self.age_days, 1),
        }


@dataclass
class AuditReport:
    """Rapport d'audit complet."""

    volume_path: str
    entries: list[AuditEntry] = field(default_factory=list)
    total_volume_size: int = 0
    used_by_pipeline: int = 0

    @property
    def total_reclaimable(self) -> int:
        return sum(e.size_bytes for e in self.entries)

    def add(self, entry: AuditEntry) -> None:
        self.entries.append(entry)

    def summary_by_category(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for entry in self.entries:
            result[entry.category] = result.get(entry.category, 0) + entry.size_bytes
        return result


def audit_models_dir(models_dir: Path, report: AuditReport) -> None:
    """Verifie le dossier models/ pour les fichiers obsoletes ou inconnus."""
    if not models_dir.exists():
        log.warning("Dossier models/ absent: %s", models_dir)
        return

    qwen_dir = models_dir / "Qwen-Image-Edit-2511"

    # --- FP8 transformers (4-step + 8-step) ---
    for fp8_name in sorted(FP8_FILENAMES):
        fp8_file = models_dir / fp8_name
        if fp8_file.exists():
            report.used_by_pipeline += fp8_file.stat().st_size
            log.info("OK: %s (%s)", fp8_name, format_size(fp8_file.stat().st_size))
        else:
            log.warning("MANQUANT: %s", fp8_name)

    # --- Repo Qwen ---
    if qwen_dir.exists():
        # Verifier les safetensors dans transformer/ (obsoletes — remplaces par FP8)
        transformer_dir = qwen_dir / "transformer"
        if transformer_dir.exists():
            for f in transformer_dir.iterdir():
                if f.suffix == ".safetensors":
                    size = f.stat().st_size
                    report.add(AuditEntry(
                        path=str(f),
                        size_bytes=size,
                        category="obsolete",
                        reason="Poids transformer BF16 — remplaces par FP8 pre-quantise",
                    ))

        # Compter la taille des fichiers attendus
        for item in qwen_dir.rglob("*"):
            if item.is_file():
                rel = item.relative_to(qwen_dir)
                parts = rel.parts
                # Fichiers transformer safetensors deja traites comme obsoletes
                if parts[0] == "transformer" and item.suffix == ".safetensors":
                    continue
                report.used_by_pipeline += item.stat().st_size
    else:
        log.warning("MANQUANT: Qwen-Image-Edit-2511/")

    # --- Ancien dossier loras/ ---
    loras_dir = models_dir / "loras"
    if loras_dir.exists():
        size = dir_size(loras_dir)
        report.add(AuditEntry(
            path=str(loras_dir),
            size_bytes=size,
            category="obsolete",
            reason="Dossier LoRA — remplace par FP8 pre-quantise avec Lightning fuse",
        ))

    # --- Anciens fichiers FP8 (renommes upstream) ---
    for old_name in OBSOLETE_FP8_FILENAMES:
        old_file = models_dir / old_name
        if old_file.exists():
            size = old_file.stat().st_size
            report.add(AuditEntry(
                path=str(old_file),
                size_bytes=size,
                category="obsolete",
                reason="Ancien fichier FP8 renomme — remplace par les checkpoints 4-step/8-step",
            ))

    # --- Tout autre fichier/dossier dans models/ ---
    known_names = {"Qwen-Image-Edit-2511", "loras"} | FP8_FILENAMES | OBSOLETE_FP8_FILENAMES
    for item in models_dir.iterdir():
        name = item.name
        if name in known_names:
            continue  # deja traite
        size = item.stat().st_size if item.is_file() else dir_size(item)
        report.add(AuditEntry(
            path=str(item),
            size_bytes=size,
            category="unknown",
            reason=f"Fichier/dossier inconnu dans models/ — non utilise par le pipeline",
        ))


def audit_outputs(output_dir: Path, max_age_days: float, report: AuditReport) -> None:
    """Identifie les outputs anciens depassant max_age_days."""
    if not output_dir.exists():
        return

    for job_dir in output_dir.iterdir():
        if not job_dir.is_dir():
            continue
        age = file_age_days(job_dir)
        size = dir_size(job_dir)
        if age > max_age_days:
            report.add(AuditEntry(
                path=str(job_dir),
                size_bytes=size,
                category="stale_output",
                reason=f"Output de {age:.0f} jours (seuil: {max_age_days:.0f}j)",
                age_days=age,
            ))
        else:
            report.used_by_pipeline += size


def audit_cache(cache_dir: Path, report: AuditReport) -> None:
    """Verifie le dossier cache/ — le cache dedup est conserve integralement."""
    if not cache_dir.exists():
        return

    dedup_dir = cache_dir / "dedup"
    if dedup_dir.exists():
        # Tout le cache dedup est fonctionnel — compter comme utilise
        report.used_by_pipeline += dir_size(dedup_dir)

    # Fichiers hors de dedup/ → orphelins
    for item in cache_dir.iterdir():
        if item.name == "dedup":
            continue
        size = item.stat().st_size if item.is_file() else dir_size(item)
        report.add(AuditEntry(
            path=str(item),
            size_bytes=size,
            category="orphan",
            reason="Fichier dans cache/ hors du dossier dedup/",
        ))


def audit_root(volume: Path, report: AuditReport) -> None:
    """Identifie les fichiers/dossiers inconnus a la racine du volume."""
    if not volume.exists():
        return

    for item in volume.iterdir():
        name = item.name
        if name in FUNCTIONAL_DIRS:
            continue  # traites par les fonctions specialisees
        # Fichiers/dossiers inconnus a la racine
        size = item.stat().st_size if item.is_file() else dir_size(item)
        report.add(AuditEntry(
            path=str(item),
            size_bytes=size,
            category="unknown",
            reason="Element inconnu a la racine du volume — non utilise par le pipeline",
        ))


def run_audit(volume_path: str, max_age_days: float) -> AuditReport:
    """Execute l'audit complet du volume."""
    volume = Path(volume_path)
    report = AuditReport(volume_path=volume_path)

    if not volume.exists():
        log.error("Volume introuvable: %s", volume_path)
        return report

    report.total_volume_size = dir_size(volume)

    log.info("Scan du volume: %s (%s)", volume_path, format_size(report.total_volume_size))
    log.info("Seuil d'anciennete: %.0f jours", max_age_days)
    log.info("")

    audit_models_dir(volume / "models", report)
    audit_outputs(volume / "output", max_age_days, report)
    audit_cache(volume / "cache", report)
    audit_root(volume, report)

    return report


def print_report(report: AuditReport, as_json: bool = False) -> None:
    """Affiche le rapport d'audit."""
    if as_json:
        data = {
            "volume_path": report.volume_path,
            "total_volume_size": report.total_volume_size,
            "total_volume_size_human": format_size(report.total_volume_size),
            "used_by_pipeline": report.used_by_pipeline,
            "used_by_pipeline_human": format_size(report.used_by_pipeline),
            "total_reclaimable": report.total_reclaimable,
            "total_reclaimable_human": format_size(report.total_reclaimable),
            "entries": [e.to_dict() for e in report.entries],
            "summary": {
                k: format_size(v)
                for k, v in report.summary_by_category().items()
            },
        }
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return

    # --- Affichage texte ---
    print()
    print("=" * 60)
    print("  AUDIT VOLUME MEDUSA I2I")
    print("=" * 60)
    print(f"  Volume          : {report.volume_path}")
    print(f"  Taille totale   : {format_size(report.total_volume_size)}")
    print(f"  Utilise pipeline: {format_size(report.used_by_pipeline)}")
    print(f"  Recuperable     : {format_size(report.total_reclaimable)}")
    print("=" * 60)

    if not report.entries:
        print("\n  Aucun fichier non utilise detecte.")
        print()
        return

    # Grouper par categorie
    categories = {
        "obsolete": "FICHIERS OBSOLETES (ancienne architecture)",
        "stale_output": "OUTPUTS ANCIENS",
        "orphan": "FICHIERS ORPHELINS",
        "unknown": "FICHIERS INCONNUS",
    }

    for cat_key, cat_label in categories.items():
        entries = [e for e in report.entries if e.category == cat_key]
        if not entries:
            continue
        cat_total = sum(e.size_bytes for e in entries)
        print(f"\n  --- {cat_label} ({format_size(cat_total)}) ---")
        for entry in sorted(entries, key=lambda e: -e.size_bytes):
            age_str = f" [{entry.age_days:.0f}j]" if entry.age_days > 0 else ""
            print(f"    {entry.size_human:>10}  {entry.path}{age_str}")
            print(f"              {entry.reason}")

    print()
    print("-" * 60)
    summary = report.summary_by_category()
    for cat_key, cat_label in categories.items():
        if cat_key in summary:
            print(f"  {cat_label:40} {format_size(summary[cat_key]):>10}")
    print("-" * 60)
    print(f"  {'TOTAL RECUPERABLE':40} {format_size(report.total_reclaimable):>10}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit du volume RunPod pour le pipeline Medusa I2I",
    )
    parser.add_argument(
        "--volume",
        default=os.environ.get("VOLUME_ROOT", "/runpod-volume"),
        help="Chemin du volume a scanner (defaut: /runpod-volume)",
    )
    parser.add_argument(
        "--max-age-days",
        type=float,
        default=30,
        help="Seuil en jours pour les outputs/caches anciens (defaut: 30)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Sortie au format JSON",
    )
    args = parser.parse_args()

    report = run_audit(args.volume, args.max_age_days)
    print_report(report, as_json=args.json)

    # Code retour non-zero si des fichiers reclaimables sont trouves
    sys.exit(1 if report.entries else 0)


if __name__ == "__main__":
    main()
