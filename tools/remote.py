#!/usr/bin/env python3
"""Sync the current repo to a remote host and run commands there.

`sync` pushes local->remote (mirror with --delete) by default; pass --pull
for the reverse (remote->local, newer-wins, no deletes).

Reads `.remote.toml` (preferred, gitignored) or `remote.toml` from the repo
root, or $REMOTE_CONFIG. Each entry under [hosts.<name>] defines `host`,
`dest`, optional `post_sync`, `excludes`, `includes`, and an optional
`[hosts.<name>.tasks]` table of named commands. A top-level `[tasks]` table
defines tasks shared across all hosts; per-host entries override on collision.

`includes` are emitted before `excludes` so a whitelist can survive a broad
exclude (rsync filter rules are first-match-wins), and each include's ancestor
directories are auto-gated so rsync recurses into them. To ignore everything
under `out/` except `out/winner`, just write the two patterns you care about:

    includes = ["out/winner/***"]
    excludes = ["out/**"]
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable

# Needs Python 3.11+ (stdlib tomllib). Shells often resolve an older
# python3; rather than fail with a confusing ModuleNotFoundError, re-exec
# under a newer interpreter if one is on PATH.
if sys.version_info < (3, 11):  # noqa: UP036
    for _cand in ("python3.13", "python3.12", "python3.11"):
        _exe = shutil.which(_cand)
        if _exe:
            os.execv(_exe, [_exe, *sys.argv])
    sys.exit(
        f"error: remote.py needs Python >= 3.11 (for tomllib); running under "
        f"{sys.version.split()[0]} and no python3.11+ found on PATH"
    )

import tomllib
from pathlib import Path
from typing import Any, cast

DEFAULT_EXCLUDES = [
    ".git/",
    ".venv/",
    "__pycache__/",
    ".mypy_cache/",
    ".ruff_cache/",
    ".pytest_cache/",
    "*.pyc",
]
PUSH_STATE_EXCLUDES = ["out/*/results.csv"]
PULL_STATE_INCLUDES = ["out/*/results.csv"]

Entry = dict[str, Any]  # pyright: ignore[reportExplicitAny]
Config = dict[str, Any]  # pyright: ignore[reportExplicitAny]


def find_config() -> Path:
    if env := os.environ.get("REMOTE_CONFIG"):
        p = Path(env).resolve()
        if not p.is_file():
            sys.exit(f"error: REMOTE_CONFIG={env} does not exist")
        return p
    cwd = Path.cwd().resolve()
    for d in [cwd, *cwd.parents]:
        for name in (".remote.toml", "remote.toml"):
            p = d / name
            if p.is_file():
                return p
    sys.exit(
        "error: no .remote.toml or remote.toml found in cwd or any parent (set REMOTE_CONFIG to override)"
    )


def load(path: Path) -> Config:
    with path.open("rb") as f:
        return tomllib.load(f)


def resolve(cfg: Config, name: str | None) -> tuple[str, Entry]:
    hosts: dict[str, Entry] = cfg.get("hosts") or {}
    if not hosts:
        sys.exit("error: remote.toml has no [hosts.*] entries")
    name = name or cast("str | None", cfg.get("default"))
    if not name:
        sys.exit(
            f"error: no config specified and `default` is unset (known: {', '.join(hosts)})"
        )
    if name not in hosts:
        sys.exit(
            f"error: config '{name}' not in remote.toml (known: {', '.join(hosts)})"
        )
    entry = hosts[name]
    for k in ("host", "dest"):
        if not entry.get(k):
            sys.exit(f"error: hosts.{name}.{k} is required")
    return name, entry


def expand_includes(includes: list[str]) -> list[str]:
    """Prefix each include with gates for its ancestor directories.

    rsync prunes a directory the moment it matches an exclude, and never
    recurses below it, so a whitelist like "out/winner/***" is dead unless
    every parent ("out/") is itself included first. That gate is pure path
    arithmetic, so derive it rather than making the user hand-write it:
    "out/winner/***" expands to ["out/", "out/winner/", "out/winner/***"].
    Results are deduped with order preserved (shared ancestors emit once).
    """
    out: list[str] = []
    seen: set[str] = set()

    def add(pat: str) -> None:
        if pat not in seen:
            seen.add(pat)
            out.append(pat)

    for pat in includes:
        anchor = "/" if pat.startswith("/") else ""
        segs = pat.lstrip("/").split("/")
        for i in range(1, len(segs)):  # every ancestor dir, shallow->deep
            add(anchor + "/".join(segs[:i]) + "/")
        add(pat)
    return out


def rsync_argv(root: Path, entry: Entry, extra: list[str], *, pull: bool) -> list[str]:
    argv = ["rsync", "-avz"]
    # Pull is newer-wins, no deletes (remote-produced files land locally without
    # clobbering local edits). Push mirrors local onto remote.
    argv += ["-u"] if pull else ["--delete"]
    if pull:
        for pat in expand_includes(PULL_STATE_INCLUDES):
            argv += ["--include", pat]
    else:
        for pat in PUSH_STATE_EXCLUDES:
            argv += ["--exclude", pat]
    # rsync filter rules are first-match-wins in argv order, so includes must
    # precede excludes: that lets a whitelist (e.g. "out/winner/***") survive a
    # broad exclude (e.g. "out/**"). expand_includes() also opens the parent-dir
    # gates the whitelist needs. See the module docstring for the pattern.
    includes = cast(list[str], entry.get("includes", []))
    for pat in expand_includes(includes):
        argv += ["--include", pat]
    excludes = cast(list[str], entry.get("excludes", DEFAULT_EXCLUDES))
    for pat in excludes:
        argv += ["--exclude", pat]
    argv += extra
    local = f"{root}/"
    remote = f"{entry['host']}:{entry['dest']}/"
    argv += [remote, local] if pull else [local, remote]
    return argv


def remote_command(entry: Entry, user_cmd: str) -> str:
    parts = [f"cd {shlex.quote(cast(str, entry['dest']))}"]
    if post := cast("str | None", entry.get("post_sync")):
        parts.append(post)
    parts.append(user_cmd)
    return " && ".join(parts)


def run(argv: list[str]) -> int:
    print("→", " ".join(shlex.quote(a) for a in argv), file=sys.stderr)
    return subprocess.run(argv).returncode


def merged_tasks(cfg: Config, entry: Entry) -> dict[str, str]:
    """Per-host tasks overlay top-level [tasks] (host wins on key collision)."""
    tasks: dict[str, str] = {}
    tasks.update(cast(dict[str, str], cfg.get("tasks") or {}))
    tasks.update(cast(dict[str, str], entry.get("tasks") or {}))
    return tasks


def cmd_info(args: argparse.Namespace, cfg: Config, _root: Path) -> int:
    hosts = cast(dict[str, Entry], cfg.get("hosts") or {})
    default = cast("str | None", cfg.get("default"))
    global_tasks = cast(dict[str, str], cfg.get("tasks") or {})
    config_name = cast("str | None", args.config)
    if config_name:
        name, entry = resolve(cfg, config_name)
        targets: list[tuple[str, Entry]] = [(name, entry)]
    else:
        targets = list(hosts.items())
    print(f"config: {find_config()}")
    print(f"default: {default or '(none)'}")
    if global_tasks:
        print("\n[tasks] (shared)")
        for tname, tcmd in global_tasks.items():
            print(f"  {tname}: {tcmd}")
    for name, entry in targets:
        marker = " (default)" if name == default else ""
        print(f"\n[{name}]{marker}")
        print(f"  host: {entry.get('host')}")
        print(f"  dest: {entry.get('dest')}")
        if post := entry.get("post_sync"):
            print(f"  post_sync: {post}")
        host_tasks = cast(dict[str, str], entry.get("tasks") or {})
        if host_tasks:
            print("  tasks:")
            for tname, tcmd in host_tasks.items():
                override = " (overrides shared)" if tname in global_tasks else ""
                print(f"    {tname}: {tcmd}{override}")
    return 0


def cmd_sync(args: argparse.Namespace, cfg: Config, root: Path) -> int:
    _, entry = resolve(cfg, cast("str | None", args.config))
    extra = cast(list[str], args.rsync_args)
    return run(rsync_argv(root, entry, extra, pull=cast(bool, args.pull)))


def cmd_run(args: argparse.Namespace, cfg: Config, _root: Path) -> int:
    _, entry = resolve(cfg, cast("str | None", args.config))
    cmd = cast(list[str], args.cmd)
    if not cmd:
        sys.exit("error: no command given")
    user_cmd = " ".join(shlex.quote(c) for c in cmd)
    return run(
        [
            "ssh",
            entry["host"],
            f"bash -lc {shlex.quote(remote_command(entry, user_cmd))}",
        ]
    )


def cmd_task(args: argparse.Namespace, cfg: Config, _root: Path) -> int:
    _, entry = resolve(cfg, cast("str | None", args.config))
    tasks = merged_tasks(cfg, entry)
    task_name = cast(str, args.task)
    if task_name not in tasks:
        sys.exit(
            f"error: task '{task_name}' not in this config (known: {', '.join(tasks) or '(none)'})"
        )
    base = tasks[task_name]
    user_cmd = base
    task_args = cast(list[str], args.task_args)
    if task_args:
        user_cmd = base + " " + " ".join(shlex.quote(a) for a in task_args)
    return run(
        [
            "ssh",
            entry["host"],
            f"bash -lc {shlex.quote(remote_command(entry, user_cmd))}",
        ]
    )


def main() -> int:
    doc = __doc__ or ""
    p = argparse.ArgumentParser(prog="remote", description=doc.splitlines()[0])
    _ = p.add_argument(
        "-c", "--config", help="config name (default: `default` field in remote.toml)"
    )
    sub = p.add_subparsers(dest="action", required=True)

    _ = sub.add_parser("info", help="print resolved config and available tasks")

    sp_sync = sub.add_parser(
        "sync",
        help="rsync local->remote (mirror); use --pull for remote->local (newer-wins)",
    )
    _ = sp_sync.add_argument(
        "--pull",
        action="store_true",
        help="pull remote->local (newer-wins, no deletes) instead of pushing",
    )
    _ = sp_sync.add_argument(
        "rsync_args",
        nargs=argparse.REMAINDER,
        help="extra rsync flags (prefix with --, e.g. -- -n)",
    )

    sp_run = sub.add_parser(
        "run",
        help="run an arbitrary command on the remote (does not sync; run `sync` first)",
    )
    _ = sp_run.add_argument(
        "cmd", nargs=argparse.REMAINDER, help="command + args (prefix with --)"
    )

    sp_task = sub.add_parser(
        "task",
        help="run a named task from remote.toml (does not sync; run `sync` first)",
    )
    _ = sp_task.add_argument("task")
    _ = sp_task.add_argument("task_args", nargs=argparse.REMAINDER)

    args = p.parse_args()
    # argparse REMAINDER includes a leading '--' if present; drop it.
    for attr in ("rsync_args", "cmd", "task_args"):
        if hasattr(args, attr):
            v = cast("list[str] | None", getattr(args, attr))
            if v and v[0] == "--":
                setattr(args, attr, v[1:])

    cfg_path = find_config()
    cfg = load(cfg_path)
    root = cfg_path.parent

    handlers: dict[str, Callable[[argparse.Namespace, Config, Path], int]] = {
        "info": cmd_info,
        "sync": cmd_sync,
        "run": cmd_run,
        "task": cmd_task,
    }
    return handlers[cast(str, args.action)](args, cfg, root)


if __name__ == "__main__":
    sys.exit(main())
