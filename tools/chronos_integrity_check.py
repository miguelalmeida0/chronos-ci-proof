#!/usr/bin/env python3
"""Enforce the installed group-isolation guardrail for the imported experiment."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

SOURCE = Path("src/injury_risk/models/train.py")
POLICY_ID = "chronos-athlete-group-isolation-v1"


def _call_name(node: ast.Call) -> str:
    function = node.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return ""


def inspect_source(source: str) -> list[dict[str, object]]:
    tree = ast.parse(source)
    findings: list[dict[str, object]] = []
    imported: set[str] = set()
    calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "sklearn.model_selection":
            imported.update(alias.name for alias in node.names)
        if isinstance(node, ast.Call):
            calls.append(node)

    for forbidden in ("StratifiedKFold", "train_test_split"):
        if forbidden in imported:
            findings.append(
                {
                    "code": "ENTITY_SPLIT_NOT_GROUP_AWARE",
                    "line": 1,
                    "message": f"{forbidden} permits athlete overlap across evaluation boundaries",
                }
            )

    if "StratifiedGroupKFold" not in imported:
        findings.append(
            {
                "code": "GROUP_SPLITTER_MISSING",
                "line": 1,
                "message": "StratifiedGroupKFold is required for repeated athlete observations",
            }
        )

    cross_validate_calls = [call for call in calls if _call_name(call) == "cross_validate"]
    if not cross_validate_calls or any(
        not any(keyword.arg == "groups" for keyword in call.keywords)
        for call in cross_validate_calls
    ):
        findings.append(
            {
                "code": "CV_GROUPS_MISSING",
                "line": next((call.lineno for call in cross_validate_calls), 1),
                "message": "cross_validate must receive the imported experiment's athlete groups",
            }
        )

    split_calls = [call for call in calls if _call_name(call) == "split"]
    if not split_calls or any(len(call.args) < 3 for call in split_calls):
        findings.append(
            {
                "code": "HOLDOUT_GROUPS_MISSING",
                "line": next((call.lineno for call in split_calls), 1),
                "message": "holdout construction must split with the athlete group identity",
            }
        )
    return findings


def run(path: Path = SOURCE) -> dict[str, object]:
    source = path.read_text(encoding="utf-8")
    findings = inspect_source(source)
    return {
        "schema_version": "1.0",
        "policy_id": POLICY_ID,
        "source": str(path),
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "conclusion": "failure" if findings else "success",
        "findings": findings,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--json-output", type=Path, default=Path("chronos-check.json"))
    args = parser.parse_args()
    result = run(args.source)
    args.json_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for finding in result["findings"]:
        print(
            f"::error file={args.source},line={finding['line']},title={finding['code']}::"
            f"{finding['message']}"
        )
    print(json.dumps(result, sort_keys=True))
    return 1 if result["findings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
