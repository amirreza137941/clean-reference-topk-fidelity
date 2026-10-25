"""Regenerate release file hashes after an intentional repository change."""

from __future__ import annotations

import hashlib

from audit_release import ROOT, files


def main() -> None:
    manifest = ROOT / "MANIFEST.sha256"
    paths = sorted(path for path in files() if path != manifest)
    lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(ROOT).as_posix()}"
        for path in paths
    ]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Recorded {len(lines)} release file hashes.")


if __name__ == "__main__":
    main()
