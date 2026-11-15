# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Durable per-module synchronization ownership for the existing PTOAS driver.

The producer supplies every emitted local domain and generated helper. This is
an artifact consistency contract, not an authentication format. An owned module
may only be rebuilt from its original bytes; edited PTO needs a new IR compile
and fresh synchronization qualification.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, cast

OWNED_PTO_MARKER = "// pypto.sync_owned.v1: paired ownership manifest required\n"


def source_sha256(source: str) -> str:
    """Hash exactly the UTF-8 PTO bytes consumed by the compiler."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def make_sync_ownership(source: str, domains: list[tuple[str, bool]], helpers: list[str]) -> dict[str, Any]:
    """Retain the producer's complete emitted-domain inventory."""
    owned = bool(domains) and all(owner for _, owner in domains) and not helpers
    return {
        "schema_version": 1,
        "source_sha256": source_sha256(source),
        "owned": owned,
        "domains": [{"function": name, "owned": owner} for name, owner in domains],
        "helpers": helpers,
        "ptoas_insert_sync": not owned,
    }


def validate_sync_ownership(source: str, manifest: dict[str, Any]) -> bool:
    """Check a complete module claim before suppressing downstream insertion."""
    if manifest.get("schema_version") != 1 or type(manifest.get("owned")) is not bool:
        raise ValueError("Unsupported synchronization ownership manifest")
    if not manifest["owned"]:
        if source.startswith(OWNED_PTO_MARKER):
            raise ValueError("Owned PTO requires a complete synchronization ownership manifest")
        return False
    raw_domains = manifest.get("domains")
    if (
        not isinstance(raw_domains, list)
        or not raw_domains
        or manifest.get("helpers") != []
        or manifest.get("ptoas_insert_sync") is not False
    ):
        raise ValueError("Synchronization ownership does not cover every emitted domain")
    domains = cast(list[object], raw_domains)
    names: set[str] = set()
    for item in domains:
        if not isinstance(item, dict):
            raise ValueError("Synchronization ownership domain must be an object")
        domain = cast(dict[str, object], item)
        name = domain.get("function")
        if not isinstance(name, str) or domain.get("owned") is not True:
            raise ValueError("Synchronization ownership does not cover every emitted domain")
        if name in names:
            raise ValueError("Synchronization ownership contains duplicate emitted domains")
        names.add(name)
    if manifest.get("source_sha256") != source_sha256(source):
        raise ValueError("Owned PTO was edited; recompile the original IR to requalify synchronization")
    return True


def read_sync_ownership(pto_path: Path, source: str) -> bool:
    """Read a sibling manifest; missing metadata retains ordinary delegation."""
    path = pto_path.with_suffix(".sync.json")
    if not path.is_file():
        if source.startswith(OWNED_PTO_MARKER):
            raise ValueError("Owned PTO requires its paired synchronization ownership manifest")
        return False
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Synchronization ownership manifest must be an object")
    return validate_sync_ownership(source, cast(dict[str, Any], manifest))


def owned_rebuild_flags(pto_path: Path, source: str) -> list[str] | None:
    """Restore the bounded A2/A3 owned-unit options, or require a fresh compile.

    A manifest is data, not a source of arbitrary subprocess arguments. Refuse
    options outside the qualified local target/placement interface rather than
    silently dropping an implementation-affecting compiler option.
    """
    if not read_sync_ownership(pto_path, source):
        return None
    manifest = json.loads(pto_path.with_suffix(".sync.json").read_text(encoding="utf-8"))
    raw_flags = manifest.get("compile_flags")
    if not isinstance(raw_flags, list):
        raise ValueError("Owned PTO rebuild requires recorded compiler options; recompile original IR")
    flags = cast(list[object], raw_flags)
    allowed = {"--enable-insert-sync=false", "--pto-level=level3", "--pto-arch=a2", "--pto-arch=a3"}
    if (
        len(flags) != 3
        or any(not isinstance(flag, str) or flag not in allowed for flag in flags)
        or "--enable-insert-sync=false" not in flags
        or "--pto-level=level3" not in flags
        or sum(flag in ("--pto-arch=a2", "--pto-arch=a3") for flag in flags) != 1
    ):
        raise ValueError("Owned PTO rebuild has unsupported compiler options; recompile original IR")
    return cast(list[str], flags)
