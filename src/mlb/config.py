"""Typed configuration loading. All tunables live in configs/*.yaml — never
hardcode a season, window, or seed inline in a module."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"


@dataclass(frozen=True)
class PitcherProjectionConfig:
    halflife_days: float
    shrinkage_k_batters: float
    min_prior_batters: int


@dataclass(frozen=True)
class BullpenConfig:
    halflife_days: float
    fatigue_lookback_days: int


@dataclass(frozen=True)
class TeamOffenseConfig:
    halflife_days: float
    shrinkage_k_games: float


@dataclass(frozen=True)
class BatterProjectionConfig:
    halflife_days: float
    shrinkage_k_pa: float


@dataclass(frozen=True)
class SimulationConfig:
    n_sims: int
    runs_per_team_distribution: str


@dataclass(frozen=True)
class CalibrationConfig:
    n_bins: int
    method: str


@dataclass(frozen=True)
class BacktestConfig:
    seasons: list[int]
    data_dir: Path
    random_seed: int
    pitcher_projection: PitcherProjectionConfig
    bullpen: BullpenConfig
    team_offense: TeamOffenseConfig
    batter_projection: BatterProjectionConfig
    simulation: SimulationConfig
    calibration: CalibrationConfig

    @property
    def raw_dir(self) -> Path:
        return REPO_ROOT / self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return REPO_ROOT / self.data_dir / "processed"


def load_backtest_config(path: Path | None = None) -> BacktestConfig:
    path = path or (CONFIGS_DIR / "backtest.yaml")
    raw = yaml.safe_load(path.read_text())
    return BacktestConfig(
        seasons=list(raw["seasons"]),
        data_dir=Path(raw["data_dir"]),
        random_seed=int(raw["random_seed"]),
        pitcher_projection=PitcherProjectionConfig(**raw["pitcher_projection"]),
        bullpen=BullpenConfig(**raw["bullpen"]),
        team_offense=TeamOffenseConfig(**raw["team_offense"]),
        batter_projection=BatterProjectionConfig(**raw["batter_projection"]),
        simulation=SimulationConfig(**raw["simulation"]),
        calibration=CalibrationConfig(**raw["calibration"]),
    )
