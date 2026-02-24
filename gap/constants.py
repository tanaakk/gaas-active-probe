"""
質量階層（Layer）の定義とスケール定数。

ブランド・固有名詞を排除し、質量のスケールと周波数特性のみに焦点を当てる。
"""

from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional


class Layer(Enum):
    """質量階層の4段階定義"""

    MICRO = 1   # 10^6 - 10^8: 個人・希少財
    MEDIUM = 2  # 10^9 - 10^10: 法人・拠点
    MACRO = 3   # 10^11 - 10^13: インフラ・金融
    GLOBAL = 4  # 10^14 - 10^17: 地球観測重心・理論物理


@dataclass(frozen=True)
class LayerConfig:
    """各階層のスケール範囲と周波数パラメータ"""

    layer: Layer
    mass_min: float
    mass_max: float
    base_frequency: float  # 基準周波数（Hz相当の抽象単位）
    resonance_threshold: float  # 同調判定閾値
    selection_criterion: str  # 選定条件の説明

    @property
    def mass_range(self) -> Tuple[float, float]:
        return (self.mass_min, self.mass_max)

    def contains(self, mass: float) -> bool:
        return self.mass_min <= mass <= self.mass_max


# 各階層の設定（質量のスケールと周波数特性のみ）
LAYER_CONFIGS: dict[Layer, LayerConfig] = {
    Layer.MICRO: LayerConfig(
        layer=Layer.MICRO,
        mass_min=1e6,
        mass_max=1e8,
        base_frequency=1e3,  # 高密度・高流動 → 高周波
        resonance_threshold=0.95,
        selection_criterion="高密度かつ高流動、固有の弾性係数が極めて高い",
    ),
    Layer.MEDIUM: LayerConfig(
        layer=Layer.MEDIUM,
        mass_min=1e9,
        mass_max=1e10,
        base_frequency=1e2,
        resonance_threshold=0.92,
        selection_criterion="組織的な慣性を持ち、一定の負圧を発生させ得る構造体",
    ),
    Layer.MACRO: LayerConfig(
        layer=Layer.MACRO,
        mass_min=1e11,
        mass_max=1e13,
        base_frequency=1e1,
        resonance_threshold=0.88,
        selection_criterion="広域の重力場を形成し、他球体の軌道に影響を与える大規模質量",
    ),
    Layer.GLOBAL: LayerConfig(
        layer=Layer.GLOBAL,
        mass_min=1e14,
        mass_max=1e17,
        base_frequency=1e0,
        resonance_threshold=0.85,
        selection_criterion="空間の曲率そのものを規定し、全事象の校正の基準となる観測点",
    ),
}


def get_layer_for_mass(mass: float) -> Optional[Layer]:
    """質量から適切な階層を返す。該当なしの場合は None"""
    for layer, config in LAYER_CONFIGS.items():
        if config.contains(mass):
            return layer
    return None
