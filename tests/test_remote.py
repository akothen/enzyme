import importlib.util
from pathlib import Path


def _load_remote():
    path = Path(__file__).resolve().parents[1] / "tools" / "remote.py"
    spec = importlib.util.spec_from_file_location("remote", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_push_protects_durable_results_before_user_filters(tmp_path):
    remote = _load_remote()
    entry = {
        "host": "box",
        "dest": "repo",
        "includes": ["out/***"],
        "excludes": [],
    }

    argv = remote.rsync_argv(tmp_path, entry, [], pull=False)

    result_filter = argv.index("out/*/results.csv")
    user_filter = argv.index("out/")
    assert argv[result_filter - 1] == "--exclude"
    assert result_filter < user_filter


def test_pull_can_retrieve_durable_results(tmp_path):
    remote = _load_remote()
    entry = {"host": "box", "dest": "repo", "excludes": ["out/***"]}

    argv = remote.rsync_argv(tmp_path, entry, [], pull=True)

    result_filter = argv.index("out/*/results.csv")
    user_filter = argv.index("out/***")
    assert argv[result_filter - 1] == "--include"
    assert result_filter < user_filter
