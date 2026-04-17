import os
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


DEFAULT_EMBEDDING_REPO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_RERANKER_REPO = "BAAI/bge-reranker-v2-m3"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _app_data_dir() -> Path:
    override = os.getenv("RAG_APP_HOME", "").strip()
    if override:
        return Path(override).expanduser().resolve()

    local_appdata = os.getenv("LOCALAPPDATA", "").strip()
    if local_appdata:
        return (Path(local_appdata) / "rag_app").resolve()

    return (Path.home() / ".rag_app").resolve()


def _run(command: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=check)


def _ollama_cli_available() -> bool:
    try:
        _run(["ollama", "--version"], check=True)
        return True
    except Exception:
        return False


def _ollama_api_ready(base_url: str, timeout_seconds: float = 2.0) -> bool:
    try:
        with urlopen(f"{base_url}/api/tags", timeout=timeout_seconds) as response:
            return 200 <= response.status < 300
    except URLError:
        return False
    except Exception:
        return False


def _wait_ollama_api(base_url: str, timeout_seconds: int = 90) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if _ollama_api_ready(base_url):
            return True
        time.sleep(1)
    return False


def _server_http_ready(server_url: str, timeout_seconds: float = 2.0) -> bool:
    try:
        with urlopen(f"{server_url.rstrip('/')}/openapi.json", timeout=timeout_seconds):
            return True
    except HTTPError as exc:
        return 200 <= int(getattr(exc, "code", 0)) < 500
    except URLError:
        return False
    except Exception:
        return False


def _wait_server_http(server_url: str, timeout_seconds: int = 45) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if _server_http_ready(server_url):
            return True
        time.sleep(0.5)
    return False


def _auto_open_browser(server_url: str) -> None:
    if not _bool_env("RAG_AUTO_OPEN_BROWSER", default=True):
        return

    def _worker() -> None:
        if not _wait_server_http(server_url, timeout_seconds=45):
            return
        try:
            webbrowser.open(server_url)
        except Exception:
            pass

    threading.Thread(target=_worker, name="open-browser", daemon=True).start()


def _start_ollama_if_needed(base_url: str) -> Optional[subprocess.Popen]:
    if _ollama_api_ready(base_url):
        return None

    print("[bootstrap] Ollama API is not ready, trying to start `ollama serve`...")

    creation_flags = 0
    if os.name == "nt":
        creation_flags = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        )

    process = subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creation_flags,
    )

    if not _wait_ollama_api(base_url, timeout_seconds=90):
        raise RuntimeError(
            "Ollama API did not become ready in time. Please start Ollama manually and retry."
        )
    return process


def _ensure_embedding_model(repo_id: str, local_dir: Path) -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)

    marker_files = [
        local_dir / "config.json",
        local_dir / "modules.json",
        local_dir / "sentence_bert_config.json",
    ]
    if any(path.exists() for path in marker_files):
        print(f"[bootstrap] Embedding model already exists: {local_dir}")
        return local_dir

    print(f"[bootstrap] Downloading embedding model: {repo_id}")
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError(
            "huggingface_hub is required for first-run model download. "
            "Install it in build/runtime environment."
        ) from exc

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"[bootstrap] Embedding model downloaded to: {local_dir}")
    return local_dir


def _ensure_reranker_model(repo_id: str, local_dir: Path) -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)

    marker_files = [
        local_dir / "config.json",
        local_dir / "tokenizer.json",
        local_dir / "pytorch_model.bin",
        local_dir / "model.safetensors",
    ]
    if any(path.exists() for path in marker_files):
        print(f"[bootstrap] Recall(reranker) model already exists: {local_dir}")
        return local_dir

    print(f"[bootstrap] Downloading recall(reranker) model: {repo_id}")
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError(
            "huggingface_hub is required for first-run model download. "
            "Install it in build/runtime environment."
        ) from exc

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"[bootstrap] Recall(reranker) model downloaded to: {local_dir}")
    return local_dir


def _ensure_ollama_model(model_name: str) -> None:
    print(f"[bootstrap] Pulling Ollama model: {model_name}")
    _run(["ollama", "pull", model_name], check=True)
    print("[bootstrap] Ollama model is ready")


def _configure_runtime_env(
    embedding_path: Path,
    reranker_path: Path,
    llm_provider: str,
    ollama_base_url: str,
) -> None:
    os.environ["RAG_EMBEDDING_LOCAL_ONLY"] = "true"
    os.environ["RAG_EMBEDDING_MODEL"] = str(embedding_path)
    os.environ.setdefault("RAG_EMBEDDING_CACHE_DIR", str(embedding_path.parent))

    os.environ["RAG_RERANKER_PROVIDER"] = "local"
    os.environ["RAG_RERANKER_MODEL"] = str(reranker_path)

    os.environ["RAG_LLM_PROVIDER"] = llm_provider
    if llm_provider == "ollama":
        os.environ["RAG_OLLAMA_BASE_URL"] = ollama_base_url


def main() -> None:
    app_dir = _app_data_dir()
    model_root = app_dir / "models"
    embedding_repo = os.getenv("RAG_BOOTSTRAP_EMBEDDING_REPO", DEFAULT_EMBEDDING_REPO)
    embedding_dir_name = embedding_repo.replace("/", "--")
    embedding_dir = model_root / "embedding" / embedding_dir_name

    reranker_repo = os.getenv("RAG_BOOTSTRAP_RERANKER_REPO", DEFAULT_RERANKER_REPO)
    reranker_dir_name = reranker_repo.replace("/", "--")
    reranker_dir = model_root / "reranker" / reranker_dir_name

    ollama_model = os.getenv("RAG_BOOTSTRAP_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    ollama_base_url = os.getenv("RAG_OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).rstrip(
        "/"
    )

    skip_embedding = _bool_env("RAG_BOOTSTRAP_SKIP_EMBEDDING", default=False)
    skip_reranker = _bool_env("RAG_BOOTSTRAP_SKIP_RERANKER", default=False)
    skip_ollama_pull = _bool_env("RAG_BOOTSTRAP_SKIP_OLLAMA_PULL", default=True)

    print(f"[bootstrap] App data directory: {app_dir}")
    app_dir.mkdir(parents=True, exist_ok=True)

    started_ollama_process: Optional[subprocess.Popen] = None
    try:
        if not skip_embedding:
            _ensure_embedding_model(embedding_repo, embedding_dir)
        else:
            print("[bootstrap] Skip embedding download by env flag")

        if not skip_reranker:
            _ensure_reranker_model(reranker_repo, reranker_dir)
        else:
            print("[bootstrap] Skip recall(reranker) download by env flag")

        llm_provider = "none"
        if _ollama_cli_available():
            try:
                started_ollama_process = _start_ollama_if_needed(ollama_base_url)
                llm_provider = "ollama"
            except Exception as exc:
                print(
                    "[bootstrap] Ollama service unavailable, switch to offline mode: "
                    f"{type(exc).__name__}"
                )
        else:
            print("[bootstrap] Ollama CLI not found, switch to offline mode")

        if llm_provider == "ollama":
            if not skip_ollama_pull:
                _ensure_ollama_model(ollama_model)
            else:
                print("[bootstrap] Skip Ollama pull by env flag")
        else:
            print(
                "[bootstrap] LLM disabled (offline mode). Using extractive answer mode."
            )

        _configure_runtime_env(
            embedding_dir,
            reranker_dir,
            llm_provider=llm_provider,
            ollama_base_url=ollama_base_url,
        )

        print("[bootstrap] Starting API server...")
        import uvicorn
        from api_server import app

        host = os.getenv("RAG_SERVER_HOST", "0.0.0.0")
        port = int(os.getenv("RAG_SERVER_PORT", "8000"))
        browser_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
        _auto_open_browser(f"http://{browser_host}:{port}/")
        uvicorn.run(app, host=host, port=port)
    finally:
        if started_ollama_process is not None and started_ollama_process.poll() is None:
            # If bootstrap started a local `ollama serve`, terminate it on app exit.
            started_ollama_process.terminate()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[bootstrap] Interrupted by user")
        sys.exit(130)
    except Exception as exc:
        print(f"[bootstrap] Fatal error: {exc}", file=sys.stderr)
        sys.exit(1)
