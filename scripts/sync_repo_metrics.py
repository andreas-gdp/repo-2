"""Collect delivery-health metrics from GitHub and store them in Neon Postgres.

Runs standalone (e.g. inside GitHub Actions). It determines the repository from
``GITHUB_REPOSITORY`` (set automatically by Actions), computes the current
metrics from live GitHub issue/PR data, and inserts one row into
``repo_metric_snapshots``, up-to-date raw issue/PR facts, and a handful of
recent ``github_activity_events``. Facts that were previously open but are no
longer returned by GitHub's open-item endpoints are reconciled to closed. It
stores observations only: policy terms such as "stale" and SLA status are
evaluated later by the agent using GLChat.

Environment:
    GITHUB_REPOSITORY        - "owner/name" (provided by GitHub Actions).
    GITHUB_TOKEN             - token with read access to the repo.
    GITHUB_EVENT_NAME        - triggering event (schedule/push/workflow_dispatch).
    DATABASE_URL             - Neon Postgres connection string.

Usage:
    python scripts/sync_repo_metrics.py            # collect + write
    python scripts/sync_repo_metrics.py --dry-run  # collect + print, no write

References:
    NONE
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import requests

GITHUB_API = "https://api.github.com"
REQUEST_TIMEOUT = 30
ACTIVITY_LIMIT = 12

CRITICAL_LABEL = "critical"
HIGH_PRIORITY_LABEL = "high-priority"
RELEASE_BLOCKER_LABEL = "release-blocker"


@dataclass
class Metrics:
    """Computed delivery-health metrics for a repository.

    Attributes:
        repo_full_name (str): Repository full name.
        latest_commit_sha (str): SHA of the latest commit on the default branch.
        open_issues (int): Open issues, excluding pull requests.
        critical_issues (int): Open issues labelled 'critical'.
        high_priority_issues (int): Open issues labelled 'high-priority'.
        release_blockers (int): Open issues labelled 'release-blocker'.
        open_prs (int): Open pull requests.
    """

    repo_full_name: str
    latest_commit_sha: str = ""
    open_issues: int = 0
    critical_issues: int = 0
    high_priority_issues: int = 0
    release_blockers: int = 0
    open_prs: int = 0


@dataclass
class ActivityEvent:
    """A single recent GitHub activity event.

    Attributes:
        event_time (str): ISO-8601 timestamp.
        event_type (str): 'issues', 'pull_request', or 'push'.
        action (str | None): e.g. 'opened', 'closed', 'synchronize'.
        item_type (str | None): 'issue', 'pull_request', or 'commit'.
        item_number (int | None): Issue/PR number, if applicable.
        item_title (str): Title or commit message summary.
        actor (str): Login of the actor.
        state (str | None): 'open', 'closed', or 'merged'.
        labels (list[str]): Label names exactly as returned by GitHub.
        latest_commit_sha (str): Associated commit SHA, if any.
    """

    event_time: str
    event_type: str
    action: str | None = None
    item_type: str | None = None
    item_number: int | None = None
    item_title: str = ""
    actor: str = ""
    state: str | None = None
    labels: list[str] = field(default_factory=list)
    latest_commit_sha: str = ""


class GitHubClient:
    """Minimal GitHub REST client with pagination.

    Attributes:
        repo (str): Repository full name.
        session (requests.Session): Authenticated HTTP session.
    """

    def __init__(self, repo: str, token: str) -> None:
        """Initialise the client.

        Args:
            repo (str): Repository full name, e.g. 'andreas-gdp/repo-1'.
            token (str): GitHub token with read access.
        """
        self.repo = repo
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
        )

    def _get(self, path: str, params: dict[str, Any] | None = None) -> requests.Response:
        """Perform a GET request against the GitHub API.

        Args:
            path (str): API path beginning with '/'.
            params (dict[str, Any] | None): Query parameters. Defaults to None.

        Returns:
            requests.Response: The HTTP response.

        Raises:
            RuntimeError: If the response status is not successful.
        """
        resp = self.session.get(f"{GITHUB_API}{path}", params=params, timeout=REQUEST_TIMEOUT)
        if resp.status_code >= 400:
            raise RuntimeError(f"GitHub API {resp.status_code} for {path}: {resp.text[:200]}")
        return resp

    def paginate(self, path: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Fetch all pages for a list endpoint.

        Args:
            path (str): API path beginning with '/'.
            params (dict[str, Any] | None): Query parameters. Defaults to None.

        Returns:
            list[dict[str, Any]]: All items across pages.
        """
        params = dict(params or {})
        params.setdefault("per_page", 100)
        page = 1
        items: list[dict[str, Any]] = []
        while True:
            params["page"] = page
            batch = self._get(path, params).json()
            if not batch:
                break
            items.extend(batch)
            if len(batch) < params["per_page"]:
                break
            page += 1
        return items

    def latest_commit_sha(self) -> str:
        """Return the SHA of the latest commit, or '' if none.

        Returns:
            str: Latest commit SHA (may be empty).
        """
        commits = self._get(f"/repos/{self.repo}/commits", {"per_page": 1}).json()
        return commits[0]["sha"] if commits else ""


def compute_metrics(client: GitHubClient) -> tuple[Metrics, list[dict[str, Any]], list[dict[str, Any]], list[ActivityEvent]]:
    """Compute metrics and recent activity for a repository.

    Args:
        client (GitHubClient): Authenticated GitHub client.
    Returns:
        tuple: Metrics, raw issues, raw pull requests, and recent activity events.
    """
    metrics = Metrics(repo_full_name=client.repo, latest_commit_sha=client.latest_commit_sha())

    # /issues returns both issues and PRs; PRs carry a 'pull_request' key.
    raw_issues = client.paginate(f"/repos/{client.repo}/issues", {"state": "open"})
    issues = [i for i in raw_issues if "pull_request" not in i]
    metrics.open_issues = len(issues)
    for issue in issues:
        labels = _label_names(issue)
        if CRITICAL_LABEL in labels:
            metrics.critical_issues += 1
        if HIGH_PRIORITY_LABEL in labels:
            metrics.high_priority_issues += 1
        if RELEASE_BLOCKER_LABEL in labels:
            metrics.release_blockers += 1

    prs = client.paginate(f"/repos/{client.repo}/pulls", {"state": "open"})
    metrics.open_prs = len(prs)

    events = _recent_activity(issues, prs, metrics.latest_commit_sha)
    return metrics, issues, prs, events


def _recent_activity(
    issues: list[dict[str, Any]],
    prs: list[dict[str, Any]],
    latest_sha: str,
) -> list[ActivityEvent]:
    """Build a small recent-activity feed from open issues and PRs.

    Args:
        issues (list[dict[str, Any]]): Open issues.
        prs (list[dict[str, Any]]): Open pull requests.
        latest_sha (str): Latest commit SHA.

    Returns:
        list[ActivityEvent]: The most recently updated items, newest first.
    """
    events: list[ActivityEvent] = []
    for issue in issues:
        events.append(
            ActivityEvent(
                event_time=issue.get("updated_at", ""),
                event_type="issues",
                action="updated",
                item_type="issue",
                item_number=issue.get("number"),
                item_title=issue.get("title", ""),
                actor=(issue.get("user") or {}).get("login", ""),
                state=issue.get("state"),
                labels=_label_names(issue),
                latest_commit_sha=latest_sha,
            )
        )
    for pr in prs:
        events.append(
            ActivityEvent(
                event_time=pr.get("updated_at", ""),
                event_type="pull_request",
                action="updated",
                item_type="pull_request",
                item_number=pr.get("number"),
                item_title=pr.get("title", ""),
                actor=(pr.get("user") or {}).get("login", ""),
                state=pr.get("state"),
                labels=_label_names(pr),
                latest_commit_sha=latest_sha,
            )
        )
    events.sort(key=lambda e: e.event_time, reverse=True)
    return events[:ACTIVITY_LIMIT]


def persist(
    metrics: Metrics,
    issues: list[dict[str, Any]],
    prs: list[dict[str, Any]],
    events: list[ActivityEvent],
    trigger_event: str,
    database_url: str,
) -> None:
    """Write metrics and activity to Neon Postgres.

    Ensures the repository is registered in ``tracked_repositories``, upserts
    raw issue/PR observations, then inserts one aggregate snapshot and events.

    Args:
        metrics (Metrics): Computed metrics.
        issues (list[dict[str, Any]]): Current GitHub issue payloads.
        prs (list[dict[str, Any]]): Current GitHub PR payloads.
        events (list[ActivityEvent]): Recent activity events.
        trigger_event (str): The workflow trigger event.
        database_url (str): Neon Postgres connection string.
    """
    import psycopg
    from psycopg.types.json import Json

    display_name = metrics.repo_full_name.split("/")[-1]
    with psycopg.connect(database_url) as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO tracked_repositories (repo_full_name, display_name)
            VALUES (%s, %s)
            ON CONFLICT (repo_full_name) DO NOTHING;
            """,
            (metrics.repo_full_name, display_name),
        )
        cur.execute(
            """
            INSERT INTO repo_metric_snapshots
                (repo_full_name, trigger_event, latest_commit_sha, open_issues,
                 critical_issues, high_priority_issues, release_blockers, open_prs)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
            """,
            (
                metrics.repo_full_name,
                trigger_event,
                metrics.latest_commit_sha,
                metrics.open_issues,
                metrics.critical_issues,
                metrics.high_priority_issues,
                metrics.release_blockers,
                metrics.open_prs,
            ),
        )
        _reconcile_absent_open_facts(
            cur,
            repo_full_name=metrics.repo_full_name,
            issue_numbers=[issue["number"] for issue in issues],
            pr_numbers=[pr["number"] for pr in prs],
        )
        for issue in issues:
            cur.execute(
                """
                INSERT INTO github_issue_facts
                    (repo_full_name, issue_number, title, state, author, created_at, updated_at,
                     closed_at, labels, url)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (repo_full_name, issue_number) DO UPDATE SET
                    title = EXCLUDED.title, state = EXCLUDED.state, author = EXCLUDED.author,
                    updated_at = EXCLUDED.updated_at, closed_at = EXCLUDED.closed_at,
                    labels = EXCLUDED.labels, url = EXCLUDED.url, synced_at = now();
                """,
                (
                    metrics.repo_full_name, issue["number"], issue.get("title", ""), issue.get("state", "open"),
                    (issue.get("user") or {}).get("login", ""), issue.get("created_at"), issue.get("updated_at"),
                    issue.get("closed_at"), Json(_label_names(issue)), issue.get("html_url", ""),
                ),
            )
        for pr in prs:
            cur.execute(
                """
                INSERT INTO github_pull_request_facts
                    (repo_full_name, pr_number, title, state, is_draft, author, created_at, updated_at,
                     merged_at, closed_at, labels, url)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (repo_full_name, pr_number) DO UPDATE SET
                    title = EXCLUDED.title, state = EXCLUDED.state, is_draft = EXCLUDED.is_draft,
                    author = EXCLUDED.author, updated_at = EXCLUDED.updated_at, merged_at = EXCLUDED.merged_at,
                    closed_at = EXCLUDED.closed_at, labels = EXCLUDED.labels, url = EXCLUDED.url, synced_at = now();
                """,
                (
                    metrics.repo_full_name, pr["number"], pr.get("title", ""), pr.get("state", "open"),
                    bool(pr.get("draft", False)), (pr.get("user") or {}).get("login", ""), pr.get("created_at"),
                    pr.get("updated_at"), pr.get("merged_at"), pr.get("closed_at"), Json(_label_names(pr)),
                    pr.get("html_url", ""),
                ),
            )
        for e in events:
            cur.execute(
                """
                INSERT INTO github_activity_events
                    (repo_full_name, event_time, event_type, action, item_type,
                     item_number, item_title, actor, state, labels, latest_commit_sha)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                """,
                (
                    metrics.repo_full_name,
                    e.event_time or datetime.now(timezone.utc).isoformat(),
                    e.event_type,
                    e.action,
                    e.item_type,
                    e.item_number,
                    e.item_title,
                    e.actor,
                    e.state,
                    Json(e.labels),
                    e.latest_commit_sha,
                ),
            )
        conn.commit()


def _reconcile_absent_open_facts(
    cursor: Any,
    *,
    repo_full_name: str,
    issue_numbers: list[int],
    pr_numbers: list[int],
) -> None:
    """Mark stored open facts absent from a successful GitHub scan as closed.

    The collector intentionally requests GitHub's open issues and pull requests
    only. A previously stored open item missing from that successful scan can no
    longer be presented to the agent as open. GitHub's close timestamp is not
    available from an absent payload, so the reconciliation time is retained as
    the best available observation time.
    """
    cursor.execute(
        """
        UPDATE github_issue_facts
        SET state = 'closed', closed_at = COALESCE(closed_at, now()), synced_at = now()
        WHERE repo_full_name = %s
          AND state = 'open'
          AND NOT (issue_number = ANY(%s));
        """,
        (repo_full_name, issue_numbers),
    )
    cursor.execute(
        """
        UPDATE github_pull_request_facts
        SET state = 'closed', closed_at = COALESCE(closed_at, now()), synced_at = now()
        WHERE repo_full_name = %s
          AND state = 'open'
          AND NOT (pr_number = ANY(%s));
        """,
        (repo_full_name, pr_numbers),
    )


def _label_names(item: dict[str, Any]) -> list[str]:
    """Extract lowercase label names from an issue/PR payload.

    Args:
        item (dict[str, Any]): Issue or PR payload.

    Returns:
        list[str]: Lowercase label names.
    """
    return [(label.get("name") or "").lower() for label in item.get("labels", [])]


def main() -> None:
    """Entry point: collect metrics and (unless --dry-run) persist them."""
    parser = argparse.ArgumentParser(description="Sync repo delivery metrics to Neon Postgres.")
    parser.add_argument("--dry-run", action="store_true", help="Collect and print without writing.")
    args = parser.parse_args()

    repo = os.getenv("GITHUB_REPOSITORY")
    token = os.getenv("GITHUB_TOKEN")
    if not repo or not token:
        print("ERROR: GITHUB_REPOSITORY and GITHUB_TOKEN are required.", file=sys.stderr)
        sys.exit(1)

    trigger_event = os.getenv("GITHUB_EVENT_NAME", "manual")

    client = GitHubClient(repo, token)
    metrics, issues, prs, events = compute_metrics(client)

    print(
        f"[{repo}] open_issues={metrics.open_issues} critical={metrics.critical_issues} "
        f"high_priority={metrics.high_priority_issues} release_blockers={metrics.release_blockers} "
        f"open_prs={metrics.open_prs} sha={metrics.latest_commit_sha[:7]}"
    )

    if args.dry_run:
        print(f"[dry-run] would insert 1 snapshot and {len(events)} activity event(s).")
        return

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL is required to persist metrics.", file=sys.stderr)
        sys.exit(1)

    persist(metrics, issues, prs, events, trigger_event, database_url)
    print(f"Upserted {len(issues)} issue(s), {len(prs)} PR(s), and inserted one snapshot for {repo}.")


if __name__ == "__main__":
    main()
