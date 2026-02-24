"""
GAP — Geometric Active Probe

階層的質量操作システム。質量のスケールと周波数特性のみに焦点を当て、
あらゆる事象を同一の幾何学平面で操作する。
"""

from gap.core import GeometricActiveProbe
from gap.constants import Layer, LayerConfig

__all__ = [
    "GeometricActiveProbe",
    "Layer",
    "LayerConfig",
]
