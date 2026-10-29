"""Append-only V2 experiment registry.

Each candidate run appends exactly one JSON object per line to
``experiments/v2_registry.jsonl``. Records are validated against the committed
``v2_registry.schema.json`` with a small dependency-free validator (the project
does not ship ``jsonschema``). The registry is never rewritten: corrections are
new records with new IDs, per ``experiments/README.md``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any


class RegistryValidationError(ValueError):
    """Raised when a record does not satisfy the committed registry schema."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(schema: dict, root: dict) -> dict:
    if "$ref" in schema:
        ref = schema["$ref"]
        if not ref.startswith("#/"):
            raise RegistryValidationError(f"unsupported $ref: {ref}")
        node: Any = root
        for part in ref[2:].split("/"):
            node = node[part]
        return node
    return schema


_TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "null": type(None),
}


def _check_type(value: Any, declared: Any, path: str) -> None:
    names = declared if isinstance(declared, list) else [declared]
    for name in names:
        expected = _TYPES[name]
        if name == "integer" and isinstance(value, bool):
            continue  # bool is a subclass of int but is not an integer here
        if name in ("integer", "number") and isinstance(value, bool):
            continue
        if isinstance(value, expected):
            return
    raise RegistryValidationError(f"{path}: expected type {declared}")


def _validate(value: Any, schema: dict, root: dict, path: str) -> None:
    schema = _resolve(schema, root)

    if "const" in schema and value != schema["const"]:
        raise RegistryValidationError(f"{path}: must equal {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        raise RegistryValidationError(f"{path}: {value!r} not in {schema['enum']}")
    if "type" in schema:
        _check_type(value, schema["type"], path)

    if isinstance(value, str):
        if "pattern" in schema and not re.search(schema["pattern"], value):
            raise RegistryValidationError(f"{path}: does not match {schema['pattern']}")
        if "minLength" in schema and len(value) < schema["minLength"]:
            raise RegistryValidationError(f"{path}: shorter than {schema['minLength']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise RegistryValidationError(f"{path}: below minimum {schema['minimum']}")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            raise RegistryValidationError(f"{path}: fewer than {schema['minItems']} items")
        if schema.get("uniqueItems") and len(value) != len(
            {json.dumps(v, sort_keys=True) for v in value}
        ):
            raise RegistryValidationError(f"{path}: items must be unique")
        item_schema = schema.get("items")
        if item_schema is not None:
            for i, item in enumerate(value):
                _validate(item, item_schema, root, f"{path}[{i}]")
    if isinstance(value, dict):
        if "minProperties" in schema and len(value) < schema["minProperties"]:
            raise RegistryValidationError(f"{path}: fewer than {schema['minProperties']} keys")
        for key in schema.get("required", []):
            if key not in value:
                raise RegistryValidationError(f"{path}: missing required key '{key}'")
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        for key, item in value.items():
            if key in properties:
                _validate(item, properties[key], root, f"{path}.{key}")
            elif additional is False:
                raise RegistryValidationError(f"{path}: unexpected key '{key}'")
            elif isinstance(additional, dict):
                _validate(item, additional, root, f"{path}.{key}")


def load_schema(schema_path: str | Path) -> dict:
    return json.loads(Path(schema_path).read_text(encoding="utf-8"))


def validate_record(record: dict, schema: dict) -> None:
    """Validate one record against the committed schema; raise on any failure."""

    _validate(record, schema, schema, "record")
    # Belt-and-braces on the two invariants that must never regress silently.
    if record.get("known_period_accessed") is not False:
        raise RegistryValidationError("known_period_accessed must be exactly false")


def load_records(registry_path: str | Path) -> list[dict]:
    path = Path(registry_path)
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def append_record(
    record: dict,
    *,
    registry_path: str | Path,
    schema_path: str | Path,
) -> None:
    """Validate then append one record. Never rewrites existing lines."""

    schema = load_schema(schema_path)
    validate_record(record, schema)

    existing = load_records(registry_path)
    ids = {r.get("experiment_id") for r in existing}
    if record["experiment_id"] in ids:
        raise RegistryValidationError(
            f"experiment_id already registered: {record['experiment_id']}"
        )

    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
