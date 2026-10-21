#!/usr/bin/env python3
"""List or delete saved W&B project workspace views via the GraphQL ``deleteView`` mutation.

``wandb-workspaces`` only exposes ``save`` / ``save_as_new_view`` (``upsertView``). Deleting
duplicate Pathology lab tabs requires this script or the UI (view menu → Delete view).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "tutorials"))

import wandb
from wandb_workspaces._graphql import execute_graphql

from pathologies.instrumentation.wandb_workspace import DEFAULT_PROJECT  # noqa: E402

_LIST_VIEWS = """
query ListProjectViews($entityName: String!, $name: String!, $first: Int!) {
  project(name: $name, entityName: $entityName) {
    allViews(viewType: "project-view", first: $first) {
      edges {
        node {
          id
          name
          displayName
          createdAt
        }
      }
    }
  }
}
"""

_DELETE_VIEW = """
mutation DeleteView($id: ID!) {
  deleteView(input: {id: $id, deleteDrafts: true}) {
    success
    pendingDrafts
  }
}
"""


def _list_project_views(
    api: wandb.Api,
    entity: str,
    project: str,
    *,
    limit: int,
) -> list[dict]:
    response = execute_graphql(
        api,
        _LIST_VIEWS,
        {"entityName": entity, "name": project, "first": limit},
    )
    project_node = response.get("project")
    if project_node is None:
        msg = f"Project not found: {entity}/{project}"
        raise RuntimeError(msg)
    edges = project_node.get("allViews", {}).get("edges", [])
    return [edge["node"] for edge in edges]


def _delete_view(api: wandb.Api, view_id: str) -> bool:
    response = execute_graphql(api, _DELETE_VIEW, {"id": view_id})
    payload = response.get("deleteView") or {}
    return bool(payload.get("success"))


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Delete duplicate W&B workspace (project) views.")
    parser.add_argument("--entity", default=None, help="W&B entity (default: WANDB_ENTITY).")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help=f"Project (default: {DEFAULT_PROJECT}).")
    parser.add_argument(
        "--display-name",
        default=None,
        help='Only delete views whose displayName matches (e.g. "Pathology lab").',
    )
    parser.add_argument(
        "--keep-latest",
        action="store_true",
        help="When filtering by display name, keep the newest view and delete the rest.",
    )
    parser.add_argument(
        "--all-project-views",
        action="store_true",
        help="Delete every project-view in the project (destructive).",
    )
    parser.add_argument("--limit", type=int, default=100, help="Max views to fetch (default: 100).")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be deleted without calling deleteView.",
    )
    args = parser.parse_args()

    entity = args.entity or os.environ.get("WANDB_ENTITY")
    if not entity:
        msg = "Set WANDB_ENTITY or pass --entity"
        raise SystemExit(msg)

    api = wandb.Api()
    views = _list_project_views(api, entity, args.project, limit=args.limit)
    if not views:
        print("No project views found.")
        return

    targets = views
    if args.all_project_views:
        targets = views
    elif args.display_name:
        targets = [v for v in views if v.get("displayName") == args.display_name]
        if args.keep_latest and targets:
            targets = sorted(targets, key=lambda v: v.get("createdAt") or "", reverse=True)[1:]
    else:
        parser.error("Specify --display-name or --all-project-views")

    if not targets:
        print("No matching views to delete.")
        return

    print(f"{'Would delete' if args.dry_run else 'Deleting'} {len(targets)} view(s) in {entity}/{args.project}:")
    for view in targets:
        print(f"  - {view.get('displayName')}  id={view['id']}  name={view.get('name')}")
        if args.dry_run:
            continue
        if not _delete_view(api, view["id"]):
            print(f"    FAILED to delete {view['id']}", file=sys.stderr)

    if args.dry_run:
        print("Dry run only. Re-run without --dry-run to delete.")


if __name__ == "__main__":
    main()
