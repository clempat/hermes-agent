from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb


@pytest.fixture
def board(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(kb.time, "time", lambda: 1_000)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    with kb.connect_closing() as conn:
        yield conn, tmp_path


def _blocked(conn, kind: str | None, *, workspace_kind="scratch", workspace_path=None):
    task_id = kb.create_task(
        conn,
        title="fixture blocker",
        assignee="worker",
        workspace_kind=workspace_kind,
        workspace_path=workspace_path,
    )
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (task_id,))
    assert kb.block_task(conn, task_id, reason="fixture", kind=kind)
    return task_id


def _nudge(conn, task_id, **overrides):
    args = {
        "signal": "GET /health returned 200",
        "rule": "service-health",
        "reason": "dependency is reachable again",
        "resolved": True,
        "read_back": "HTTP 200 with expected body",
        "allowed_auto_rules": {"service-health"},
        "cooldown_seconds": 3600,
    }
    args.update(overrides)
    return kb.controlled_nudge(conn, task_id, **args)


def test_allowlisted_resolved_transient_is_auto_unblocked_with_audit(board):
    conn, _ = board
    task_id = _blocked(conn, "transient")

    result = _nudge(conn, task_id)

    assert result == {"action": "unblocked", "old_status": "blocked", "new_status": "ready"}
    assert kb.get_task(conn, task_id).status == "ready"
    event = kb.list_events(conn, task_id)[-1]
    assert event.kind == "auto_unblocked"
    assert event.payload == {
        "signal": "GET /health returned 200",
        "rule": "service-health",
        "old_status": "blocked",
        "new_status": "ready",
        "reason": "dependency is reachable again",
        "read_back": "HTTP 200 with expected body",
    }
    assert kb.list_comments(conn, task_id)[-1].body.startswith("AUTO-UNBLOCK:")


def test_needs_input_is_only_nudged_even_with_positive_readback(board):
    conn, _ = board
    task_id = _blocked(conn, "needs_input")

    result = _nudge(conn, task_id)

    assert result["action"] == "nudged"
    assert kb.get_task(conn, task_id).status == "blocked"
    event = kb.list_events(conn, task_id)[-1]
    assert event.kind == "blocker_nudged"
    assert event.payload["reason"] == "human blocker cannot be auto-unblocked"


def test_resolved_transient_with_open_child_is_only_nudged(board):
    conn, _ = board
    task_id = _blocked(conn, "transient")
    child = kb.create_task(conn, title="dangerous child", assignee="worker", parents=[task_id])

    result = _nudge(conn, task_id)

    assert result["action"] == "nudged"
    assert kb.get_task(conn, task_id).status == "blocked"
    assert kb.get_task(conn, child).status == "todo"
    assert kb.list_events(conn, task_id)[-1].payload["reason"] == "open children require review"


def test_invalid_worktree_is_only_nudged(board):
    conn, _ = board
    task_id = _blocked(conn, "transient", workspace_kind="worktree")

    result = _nudge(conn, task_id)

    assert result["action"] == "nudged"
    assert kb.get_task(conn, task_id).status == "blocked"
    assert kb.list_events(conn, task_id)[-1].payload["reason"] == "worktree workspace is not verified"


def test_fake_git_marker_does_not_verify_worktree(board):
    conn, tmp_path = board
    fake = tmp_path / "fake-worktree"
    fake.mkdir()
    (fake / ".git").write_text("not a gitdir", encoding="utf-8")
    task_id = _blocked(
        conn,
        "transient",
        workspace_kind="worktree",
        workspace_path=str(fake),
    )

    result = _nudge(conn, task_id)

    assert result["action"] == "nudged"
    assert kb.get_task(conn, task_id).status == "blocked"
    assert kb.list_events(conn, task_id)[-1].payload["reason"] == "worktree workspace is not verified"


def test_worktree_verification_ignores_inherited_git_overrides(board, monkeypatch):
    _, tmp_path = board
    repo = tmp_path / "repo"
    linked = tmp_path / "linked"
    fake = tmp_path / "fake"
    repo.mkdir()
    fake.mkdir()
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Test"], check=True
    )
    (repo / "file").write_text("fixture", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "file"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "fixture"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "worktree", "add", "-b", "linked", str(linked)],
        check=True,
        capture_output=True,
    )
    assert kb._verified_git_worktree(str(linked))

    git_dir = subprocess.run(
        ["git", "-C", str(linked), "rev-parse", "--path-format=absolute", "--git-dir"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    common_dir = subprocess.run(
        [
            "git",
            "-C",
            str(linked),
            "rev-parse",
            "--path-format=absolute",
            "--git-common-dir",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.setenv("GIT_DIR", git_dir)
    monkeypatch.setenv("GIT_COMMON_DIR", common_dir)
    monkeypatch.setenv("GIT_WORK_TREE", str(fake))

    assert not kb._verified_git_worktree(str(fake))


def test_nudge_is_deduplicated_until_cooldown_expires(board, monkeypatch):
    conn, _ = board
    task_id = _blocked(conn, "capability")

    first = _nudge(conn, task_id, resolved=False, read_back=None)
    second = _nudge(conn, task_id, resolved=False, read_back=None)

    assert first["action"] == "nudged"
    assert second["action"] == "deduplicated"
    assert len([e for e in kb.list_events(conn, task_id) if e.kind == "blocker_nudged"]) == 1

    monkeypatch.setattr(kb.time, "time", lambda: 4_601)
    third = _nudge(conn, task_id, resolved=False, read_back=None)
    assert third["action"] == "nudged"
    assert len([e for e in kb.list_events(conn, task_id) if e.kind == "blocker_nudged"]) == 2


def test_dedupe_matches_any_same_signal_in_current_block_cycle(board):
    conn, _ = board
    task_id = _blocked(conn, "capability")

    first_a = _nudge(conn, task_id, signal="A", resolved=False, read_back=None)
    signal_b = _nudge(conn, task_id, signal="B", resolved=False, read_back=None)
    second_a = _nudge(conn, task_id, signal="A", resolved=False, read_back=None)

    assert first_a["action"] == "nudged"
    assert signal_b["action"] == "nudged"
    assert second_a["action"] == "deduplicated"


def test_reblock_starts_a_new_dedupe_cycle(board):
    conn, _ = board
    task_id = _blocked(conn, "transient")
    assert _nudge(conn, task_id)["action"] == "unblocked"
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (task_id,))
    assert kb.block_task(conn, task_id, reason="new blocker", kind="capability")

    result = _nudge(conn, task_id, resolved=False, read_back=None)

    assert result["action"] == "nudged"


def test_stale_block_kind_cannot_bypass_dispatcher_circuit_breaker(board):
    conn, _ = board
    task_id = _blocked(conn, "transient")
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET consecutive_failures=2, "
            "last_failure_error='budget exhausted' WHERE id=?",
            (task_id,),
        )
        kb._append_event(conn, task_id, "gave_up", {"failures": 2})

    result = _nudge(conn, task_id, signal="new healthy signal")

    assert result["action"] == "nudged"
    task = kb.get_task(conn, task_id)
    assert task.status == "blocked"
    assert task.consecutive_failures == 2
    assert kb.list_events(conn, task_id)[-1].payload["reason"] == (
        "current blocker was not explicitly classified"
    )


def test_capability_requires_allowlist_resolution_and_readback(board):
    conn, _ = board
    task_id = _blocked(conn, "capability")

    result = _nudge(conn, task_id, read_back=None)

    assert result["action"] == "nudged"
    assert kb.get_task(conn, task_id).status == "blocked"
    assert kb.list_events(conn, task_id)[-1].payload["reason"] == "positive read-back is required"


def test_cli_nudge_uses_configured_rule_allowlist(board, monkeypatch):
    conn, _ = board
    task_id = _blocked(conn, "transient")
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"kanban": {"auto_unblock_rules": ["service-health"]}},
    )

    output = kc.run_slash(
        f"nudge {task_id} --signal 'GET /health returned 200' "
        "--rule service-health --reason 'dependency is reachable again' "
        "--resolved --read-back 'HTTP 200 with expected body'"
    )

    assert f"Auto-unblocked {task_id}" in output
    assert kb.get_task(conn, task_id).status == "ready"


def test_cli_nudge_fails_closed_without_allowlist(board, monkeypatch):
    conn, _ = board
    task_id = _blocked(conn, "transient")
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})

    output = kc.run_slash(
        f"nudge {task_id} --signal healthy --rule service-health "
        "--reason recovered --resolved --read-back verified"
    )

    assert f"Nudged {task_id}" in output
    assert kb.get_task(conn, task_id).status == "blocked"
