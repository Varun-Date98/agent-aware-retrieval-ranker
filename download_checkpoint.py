"""Download reranker checkpoint from GitHub releases on startup."""
import urllib.request
from pathlib import Path

CHECKPOINT_URL = "https://github.com/Varun-Date98/agent-aware-retrieval-ranker/releases/download/reranker-v1/reranker_best.pt"
CHECKPOINT_PATH = Path("checkpoints/reranker_best.pt")


def download_checkpoint():
    """Download checkpoint if it doesn't exist locally."""
    if CHECKPOINT_PATH.exists():
        print(f"✓ Checkpoint already exists at {CHECKPOINT_PATH}")
        return

    print(f"Downloading checkpoint from {CHECKPOINT_URL}...")
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        urllib.request.urlretrieve(CHECKPOINT_URL, CHECKPOINT_PATH)
        print(f"✓ Checkpoint downloaded successfully to {CHECKPOINT_PATH}")
        print(f"  Size: {CHECKPOINT_PATH.stat().st_size / (1024 * 1024):.2f} MB")
    except Exception as e:
        print(f"✗ Failed to download checkpoint: {e}")
        raise


if __name__ == "__main__":
    download_checkpoint()
