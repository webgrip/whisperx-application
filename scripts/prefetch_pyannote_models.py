from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

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
        local_dir_use_symlinks=False,
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

    models_dir = Path(args.models_dir)
    py_dir = models_dir / "pyannote"
    py_dir.mkdir(parents=True, exist_ok=True)

    # IMPORTANT: include 'pyannote' in file name so pyannote.audio infers correct embedding type.
    download_and_rename(SEG_REPO, py_dir / "pyannote_model_segmentation-3.0.bin", args.hf_token)
    download_and_rename(EMB_REPO, py_dir / "pyannote_model_wespeaker-voxceleb-resnet34-LM.bin", args.hf_token)

    print("Done. Ensure your config references these local .bin files.")

if __name__ == "__main__":
    main()
