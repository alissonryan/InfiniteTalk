import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


# User config
PROMPT = "A person talking."
SIZE = "infinitetalk-480"  # infinitetalk-480 | infinitetalk-720
MODE = "clip"  # clip | streaming
SAMPLE_STEPS = 40
MOTION_FRAME = 9
FRAME_NUM = 81
NUM_PERSISTENT_PARAM_IN_DIT = 0
SAVE_FILE = "colab_output"

# Optional: cache downloads on Google Drive
USE_DRIVE_CACHE = False
DRIVE_WEIGHTS_DIR = "/content/drive/MyDrive/InfiniteTalk/weights"
DRIVE_HF_CACHE_DIR = "/content/drive/MyDrive/InfiniteTalk/hf_cache"


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _pip_install(packages: list[str]) -> None:
    _run([sys.executable, "-m", "pip", "install", "-U"] + packages)


def _ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg"):
        return
    _run(["apt-get", "update"])
    _run(["apt-get", "install", "-y", "ffmpeg"])


def _maybe_mount_drive() -> None:
    if not USE_DRIVE_CACHE:
        return
    try:
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive")
    except Exception as e:
        raise RuntimeError(
            "USE_DRIVE_CACHE=True, mas não consegui montar o Google Drive."
        ) from e


def _configure_hf_cache() -> None:
    if not USE_DRIVE_CACHE:
        return
    os.makedirs(DRIVE_HF_CACHE_DIR, exist_ok=True)
    os.environ.setdefault("HF_HOME", DRIVE_HF_CACHE_DIR)
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(DRIVE_HF_CACHE_DIR, "transformers"))
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(DRIVE_HF_CACHE_DIR, "hub"))


def _ensure_hf_token() -> None:
    if os.getenv("HF_TOKEN"):
        return
    try:
        import getpass

        token = getpass.getpass("HF_TOKEN (enter para pular): ")
        if token:
            os.environ["HF_TOKEN"] = token
    except Exception:
        pass


def _download_weights(repo_root: Path) -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    weights_dir = repo_root / "weights"
    if USE_DRIVE_CACHE:
        drive_weights = Path(DRIVE_WEIGHTS_DIR)
        drive_weights.mkdir(parents=True, exist_ok=True)
        if weights_dir.exists() and weights_dir.is_symlink():
            weights_dir = weights_dir.resolve()
        elif weights_dir.exists() and any(weights_dir.iterdir()):
            print(
                f"WARNING: {weights_dir} já existe e não está vazio; usando este diretório (sem symlink pro Drive)."
            )
        else:
            if weights_dir.exists():
                weights_dir.rmdir()
            os.symlink(str(drive_weights), str(weights_dir))
            weights_dir = drive_weights
    else:
        weights_dir.mkdir(parents=True, exist_ok=True)

    token = os.getenv("HF_TOKEN")

    wan_dir = weights_dir / "Wan2.1-I2V-14B-480P"
    wav2vec_dir = weights_dir / "chinese-wav2vec2-base"
    infinitetalk_dir = weights_dir / "InfiniteTalk"

    if not wan_dir.exists() or not any(wan_dir.iterdir()):
        snapshot_download(
            repo_id="Wan-AI/Wan2.1-I2V-14B-480P",
            local_dir=str(wan_dir),
            local_dir_use_symlinks=False,
            token=token,
        )

    if not wav2vec_dir.exists() or not any(wav2vec_dir.iterdir()):
        snapshot_download(
            repo_id="TencentGameMate/chinese-wav2vec2-base",
            local_dir=str(wav2vec_dir),
            local_dir_use_symlinks=False,
            token=token,
        )
    hf_hub_download(
        repo_id="TencentGameMate/chinese-wav2vec2-base",
        filename="model.safetensors",
        revision="refs/pr/1",
        local_dir=str(wav2vec_dir),
        local_dir_use_symlinks=False,
        token=token,
    )

    if not infinitetalk_dir.exists() or not any(infinitetalk_dir.iterdir()):
        snapshot_download(
            repo_id="MeiGen-AI/InfiniteTalk",
            local_dir=str(infinitetalk_dir),
            local_dir_use_symlinks=False,
            token=token,
        )


def _colab_upload_one(prompt: str) -> Path:
    try:
        from google.colab import files  # type: ignore
    except Exception as e:
        raise RuntimeError("Este script é para rodar no Google Colab.") from e

    print(prompt)
    uploaded = files.upload()
    if len(uploaded) != 1:
        raise RuntimeError(f"Envie exatamente 1 arquivo (recebi {len(uploaded)}).")
    filename = next(iter(uploaded.keys()))
    return Path.cwd() / filename


def _ensure_wav_16k(audio_path: Path) -> Path:
    if audio_path.suffix.lower() == ".wav":
        return audio_path
    out = audio_path.with_suffix("")
    out = Path(str(out) + "_16k.wav")
    _run([
        "ffmpeg",
        "-y",
        "-i",
        str(audio_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(out),
    ])
    return out


def main() -> None:
    try:
        repo_root = Path(__file__).resolve().parents[1]
    except NameError:
        repo_root = Path.cwd()

    os.chdir(repo_root)
    print(f"Repo root: {repo_root}")

    _maybe_mount_drive()
    _configure_hf_cache()
    _ensure_hf_token()

    _ensure_ffmpeg()

    # Install deps
    _pip_install(
        [
            "torch==2.4.1",
            "torchvision==0.19.1",
            "torchaudio==2.4.1",
            "--index-url",
            "https://download.pytorch.org/whl/cu121",
        ]
    )
    _pip_install(
        [
            "xformers==0.0.28",
            "--index-url",
            "https://download.pytorch.org/whl/cu121",
        ]
    )
    _pip_install(["-r", "requirements.txt"])
    _pip_install(["librosa", "soundfile", "einops", "huggingface_hub[cli]", "misaki[en]"])

    import torch

    print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        try:
            _run(["nvidia-smi"])
        except Exception:
            pass
    else:
        print("WARNING: CUDA não está disponível. Selecione Runtime → GPU no Colab.")

    _download_weights(repo_root)

    image_path = _colab_upload_one("1) Faça upload da IMAGEM (.png/.jpg)")
    audio_path = _colab_upload_one("2) Faça upload do ÁUDIO (.wav recomendado; mp3/m4a ok)")
    audio_path = _ensure_wav_16k(audio_path)

    input_json_path = repo_root / "colab_input.json"
    with input_json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "prompt": PROMPT,
                "cond_video": str(image_path),
                "cond_audio": {"person1": str(audio_path)},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    cmd = [
        sys.executable,
        "generate_infinitetalk.py",
        "--ckpt_dir",
        "weights/Wan2.1-I2V-14B-480P",
        "--wav2vec_dir",
        "weights/chinese-wav2vec2-base",
        "--infinitetalk_dir",
        "weights/InfiniteTalk/single/infinitetalk.safetensors",
        "--input_json",
        str(input_json_path),
        "--size",
        SIZE,
        "--sample_steps",
        str(SAMPLE_STEPS),
        "--mode",
        MODE,
        "--motion_frame",
        str(MOTION_FRAME),
        "--frame_num",
        str(FRAME_NUM),
        "--num_persistent_param_in_dit",
        str(NUM_PERSISTENT_PARAM_IN_DIT),
        "--save_file",
        SAVE_FILE,
    ]
    _run(cmd)

    out_path = repo_root / f"{SAVE_FILE}.mp4"
    if out_path.exists():
        try:
            from IPython.display import Video, display  # type: ignore

            display(Video(str(out_path), embed=True))
        except Exception:
            print(f"Gerado: {out_path}")
    else:
        print("Não encontrei o vídeo de saída. Verifique os logs acima.")


if __name__ == "__main__":
    main()
