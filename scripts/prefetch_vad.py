from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET


def _get_whisperx_vad_url() -> str:
    import whisperx.vad as vad

    url = getattr(vad, "VAD_SEGMENTATION_URL", None)
    if not isinstance(url, str) or not url:
        raise RuntimeError("Could not locate whisperx.vad.VAD_SEGMENTATION_URL")
    return url


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Prefetch WhisperX VAD model into the mounted models directory. "
            "This prevents whisperx from downloading VAD at runtime."
        )
    )
    p.add_argument("--models-dir", default="models", help="Base models directory on host.")
    p.add_argument(
        "--out",
        default="vad/whisperx_vad.bin",
        help="Path under --models-dir to write the VAD blob.",
    )

    args = p.parse_args()

    models_dir = Path(args.models_dir)
    dest = models_dir / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)

    url = _get_whisperx_vad_url()

    # Use curl for robust redirect handling.
    # -f: fail on HTTP errors
    # -sS: quiet but show errors
    # -L: follow redirects
    cmd = ["curl", "-f", "-sS", "-L", "-o", str(dest), url]
    subprocess.run(cmd, check=True)

    try:
        head = dest.open("rb").read(256)
        # S3/proxies often return XML/HTML error payloads (AccessDenied, ExpiredToken, etc.)
        # which later crash in torch.load with "invalid load key, '<'".
        if head.lstrip().startswith(b"<"):
            snippet = head.decode("utf-8", errors="replace")
            code = msg = req = None
            try:
                # Best-effort parsing for S3-style errors.
                tree = ET.parse(dest)
                root = tree.getroot()
                # Tag names may be namespaced; match by suffix.
                def _find_text(suffix: str):
                    for el in root.iter():
                        if el.tag.endswith(suffix) and el.text:
                            return el.text.strip()
                    return None
                code = _find_text("Code")
                msg = _find_text("Message")
                req = _find_text("RequestId") or _find_text("RequestID")
            except Exception:
                pass
            finally:
                dest.unlink(missing_ok=True)

            details = ""
            if code or msg or req:
                details = f" (code={code!r} message={msg!r} request_id={req!r})"
            raise RuntimeError(
                "Downloaded a non-model payload (starts with '<'; likely an XML/HTML error page).\n"
                f"URL: {url}\n"
                f"Head: {snippet!r}{details}\n"
                "This is almost always a network/proxy/firewall issue on the machine/container doing the download."
            )
    except Exception as e:
        raise RuntimeError(f"VAD download produced an invalid file at {str(dest)!r}: {e}")

    h = hashlib.sha256()
    with dest.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    sha256 = h.hexdigest()

    size_bytes = dest.stat().st_size

    # Best-effort: compare against the checksum WhisperX expects for this version.
    try:
        import whisperx.vad as vad

        expected = None
        for name in ("VAD_SEGMENTATION_SHA256", "VAD_SHA256", "VAD_MODEL_SHA256"):
            val = getattr(vad, name, None)
            if isinstance(val, str) and len(val) == 64:
                expected = val
                break
        if expected and expected.lower() != sha256.lower():
            raise RuntimeError(
                "Downloaded VAD blob SHA256 does not match the WhisperX expected checksum. "
                f"expected={expected} got={sha256}"
            )
    except ImportError:
        pass

    print(f"Downloaded WhisperX VAD -> {dest}")
    print(f"VAD size -> {size_bytes} bytes")
    print(f"VAD SHA256 -> {sha256}")
    print(f"Set VAD_FILE=/app/models/{args.out} (or use .env.example)")


if __name__ == "__main__":
    main()
