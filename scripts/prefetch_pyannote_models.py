from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError

SEG_REPO = "pyannote/segmentation-3.0"
EMB_REPO = "pyannote/wespeaker-voxceleb-resnet34-LM"
FILENAME = "pytorch_model.bin"

def download_and_rename(repo_id: str, target_path: Path, token: str | None) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = hf_hub_download(
        repo_id=repo_id,
        filename=FILENAME,
        token=token,
        local_dir=str(target_path.parent),
    )
    # hf_hub_download returns the final file path already inside local_dir
    src = Path(tmp)
    # Move/rename into our convention
    src.replace(target_path)
    print(f"{repo_id}/{FILENAME} -> {target_path}")

def main() -> None:
    p = argparse.ArgumentParser(description="Prefetch gated pyannote diarization weights for offline use.")
    p.add_argument("--models-dir", default="models", help="Base models directory.")
    p.add_argument("--hf-token", default=None, help="Hugging Face token (often required for pyannote models).")
    args = p.parse_args()

    if args.hf_token is None:
        # Prefer repo's convention but also support the upstream env var.
        args.hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    if not args.hf_token:
        raise SystemExit(
            "HF_TOKEN is required for pyannote gated models. Set HF_TOKEN in .env (Compose env_file) or export it."
        )

    models_dir = Path(args.models_dir)
    py_dir = models_dir / "pyannote"
    py_dir.mkdir(parents=True, exist_ok=True)

    try:
        # IMPORTANT: include 'pyannote' in file name so pyannote.audio infers correct embedding type.
        download_and_rename(SEG_REPO, py_dir / "pyannote_model_segmentation-3.0.bin", args.hf_token)
        download_and_rename(EMB_REPO, py_dir / "pyannote_model_wespeaker-voxceleb-resnet34-LM.bin", args.hf_token)
    except GatedRepoError as e:
        raise SystemExit(
            "Pyannote models are gated on Hugging Face. Your token is missing access or you haven't accepted the terms.\n"
            f"Request/accept access here: https://huggingface.co/{SEG_REPO} and https://huggingface.co/{EMB_REPO}"
        ) from e

    print("Done. Ensure your config references these local .bin files.")

if __name__ == "__main__":
    main()
