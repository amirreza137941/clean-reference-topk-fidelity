"""Load actual original symbols without importing dataset/download entry points."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import math
import random
import sys
import types
import typing
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "experiments" / "frozen"


def original_symbols(
    filename: str, names: list[str], **extra: object
) -> types.ModuleType:
    """Compile unchanged source AST nodes from the archived user programs."""
    source = (FROZEN / filename).read_text()
    tree = ast.parse(source)
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    if {node.name for node in nodes} != set(names):
        raise AssertionError(f"missing original symbols in {filename}")
    module_name = "_parity_" + filename.removesuffix(".py")
    module = types.ModuleType(module_name)
    sys.modules[module_name] = module
    typing_names = (
        "Any",
        "Callable",
        "Dict",
        "Iterable",
        "List",
        "Mapping",
        "Optional",
        "Sequence",
        "Tuple",
    )
    module.__dict__.update({name: getattr(typing, name) for name in typing_names})
    # Recover normalization constants from original files, not the refactor.
    for original_tree in (
        ast.parse((FROZEN / "topk_cifar100_r2_common.py").read_text()),
        tree,
    ):
        for assignment in original_tree.body:
            if not isinstance(assignment, ast.Assign):
                continue
            for target in assignment.targets:
                if isinstance(target, ast.Name) and target.id.endswith(
                    ("_MEAN", "_STD")
                ):
                    try:
                        module.__dict__[target.id] = ast.literal_eval(assignment.value)
                    except (ValueError, TypeError):
                        pass  # R2 training aliases the protocol constants.
    module.__dict__.update(
        torch=torch,
        nn=nn,
        F=F,
        np=np,
        math=math,
        random=random,
        hashlib=hashlib,
        dataclass=dataclass,
        Path=Path,
        **extra,
    )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    body = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(body, str(FROZEN / filename), "exec"), module.__dict__)
    return module


@pytest.fixture(scope="session", autouse=True)
def small_cpu_thread_pool():
    """Avoid spawning many BLAS/OpenMP threads for tiny comparison models."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="session")
def original_core():
    path = FROZEN / "topk_cifar100_square_attack_core_r4.py"
    spec = importlib.util.spec_from_file_location("_original_r4_core", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def original_c10():
    base = original_symbols(
        "evaluate_final_top1_set2_defense_cifar10.py",
        ["TargetedGaussianDefense", "DefenseSpec", "averaged_actual"],
    )
    return original_symbols(
        "evaluate_topk_prior_baseline_attack_cifar10.py",
        [
            "VariableQueryOracle",
            "PerDrawQueryOracle",
            "exact_set_attack_margin",
            "per_draw_exact_set_survival_score",
            "square_reference_p_selection",
            "square_adaptive_query_plan",
            "square_racing_query_plan",
            "square_vertical_stripe_initialization",
            "square_reference_linf_proposal",
            "_archive_accepted_candidates",
            "_insert_racing_candidate",
            "square_adaptive_attack_budget",
            "square_racing_attack_budget",
        ],
        base=base,
    )
