"""Preserve tracked Como sources/environment without changing historical manifests.

Run before packaging changes: python scripts/migration/preserve_como.py
The local archive includes tracked Gym4ReaL inputs, but excludes Git metadata,
credentials, virtual environments and generated experiment outputs. Results and
checkpoints remain in their original locations and must be backed up separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from pathlib import PurePosixPath
import shutil
import subprocess
import tarfile


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args]).decode().strip()


def tracked_files(root: Path):
    names = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z"])
    for name in names.decode().split("\0"):
        if name and (root / name).is_file():
            yield root / name


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preserve(root: Path, destination: Path) -> dict:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite preservation snapshot: {destination}")
    dependency = root / "external/gym4real"
    # Inventory before creating output, so this run cannot inventory itself.
    status = git(root, "status", "--short")
    paths = sorted(set(tracked_files(root)) | set(tracked_files(dependency)))
    manifest = {
        "schema_version": 1,
        "project_revision": git(root, "rev-parse", "HEAD"),
        "project_worktree_status": status,
        "gym4real_revision": git(dependency, "rev-parse", "HEAD"),
        "gym4real_worktree_status": git(dependency, "status", "--short"),
        "original_root": str(root),
        "files_sha256": {str(p.relative_to(root)): sha256(p) for p in paths},
        "scope": "Tracked source/environment and tracked Gym4ReaL inputs; current working-tree bytes.",
        "excluded": "Ignored results/checkpoints, local figures, PDFs, Git metadata, virtual environments and untracked files.",
        "historical_manifests": "Unchanged. Absolute paths may require restoring under the original root or an explicit verification path mapping.",
    }
    destination.mkdir(parents=True)
    archive = destination / "como_source_environment.tar.gz"
    with tarfile.open(archive, "w:gz", compresslevel=1) as bundle:
        for path in paths:
            bundle.add(path, arcname=str(path.relative_to(root)), recursive=False)
    manifest["archive_sha256"] = sha256(archive)
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    verify(destination)
    return manifest


def verify(destination: Path) -> None:
    manifest = json.loads((destination / "manifest.json").read_text())
    archive = destination / "como_source_environment.tar.gz"
    if sha256(archive) != manifest["archive_sha256"]:
        raise ValueError("Snapshot archive hash mismatch")
    observed = {}
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            if not member.isfile():
                raise ValueError(f"Unexpected non-file snapshot member: {member.name}")
            member_path = PurePosixPath(member.name)
            if member_path.is_absolute() or '..' in member_path.parts:
                raise ValueError(f"Unsafe snapshot member: {member.name}")
            stream = bundle.extractfile(member)
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
            observed[member.name] = digest.hexdigest()
    if observed != manifest["files_sha256"]:
        raise ValueError("Snapshot content hash mismatch")


def restore(snapshot: Path, destination: Path) -> None:
    verify(snapshot)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Restore directory must be empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(snapshot / "como_source_environment.tar.gz", "r:gz") as bundle:
        for member in bundle:
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.extractfile(member) as source, target.open('wb') as output:
                shutil.copyfileobj(source, output)
    print(f"Restored original source and environment to {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/migration/como_before_multisystem_v1"))
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--restore-to", type=Path, help="Restore a verified snapshot into an empty directory")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    destination = args.output.resolve()
    if args.restore_to:
        restore(destination, args.restore_to.resolve())
    elif args.verify:
        verify(destination)
    else:
        preserve(root, destination)
    print(f"Verified preservation snapshot: {destination}")


if __name__ == "__main__":
    main()
