"""Helper to author notebooks from a flat list of (kind, source) cells."""
from __future__ import annotations

import json
from pathlib import Path


def nb(cells: list[tuple[str, str]]) -> dict:
    out_cells = []
    for kind, src in cells:
        if kind == "md":
            out_cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": src.splitlines(keepends=True),
            })
        elif kind == "code":
            out_cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": src.splitlines(keepends=True),
            })
        else:
            raise ValueError(kind)
    return {
        "cells": out_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write(path: Path, cells: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb(cells), indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path}")
