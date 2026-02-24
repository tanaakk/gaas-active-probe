"""
GAP — GAAS Active Probe

階層的質量操作システム。質量のスケールと周波数特性のみに焦点を当て、
あらゆる事象を同一の幾何学平面で操作する。
"""

from gap.constants import Layer, LayerConfig

try:
    from gap.core import GAASActiveProbe
    GeometricActiveProbe = GAASActiveProbe
except ImportError:
    GAASActiveProbe = None
    GeometricActiveProbe = None

__all__ = [
    "GAASActiveProbe",
    "GeometricActiveProbe",  # 後方互換
    "Layer",
    "LayerConfig",
]
