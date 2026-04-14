#!/usr/bin/env python3
"""Wrapper for openai_server_inference.py that patches mmred.const with Russian names.

This must be run instead of openai_server_inference.py when evaluating on
Russian-language benchmarks (e.g. mera_mmred_ru), so that the Pydantic
answer-validation schemas (RoomAnswer, PersonAnswer) use Russian literals.
"""

import importlib
import importlib.util
import sys
from pathlib import Path

import mmred.const as const

# Patch with Russian equivalents before the inference module builds its schemas.
const.ROOMS = ["Кухня", "Ванная", "Сад", "Офис", "Спальня", "Коридор"]
const.CHARS = ["Сандра", "Мария", "Иван", "Даниил", "Михаил"]
const.NOBODY = "Никто"

# Import the inference module by file path so we don't need scripts/__init__.py.
_inference_path = Path(__file__).parent / "openai_server_inference.py"
_spec = importlib.util.spec_from_file_location("openai_server_inference", _inference_path)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

main = _mod.main  # noqa: E402

if __name__ == "__main__":
    main()
