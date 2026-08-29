"""Reproducible multi-stage experiment campaigns."""

from .phase7_campaign import Phase7Config, run_phase7_campaign
from .phase8_campaign import Phase8Config, run_phase8_campaign
from .phase11_campaign import Phase11Config, run_phase11_campaign
from .phase12_campaign import Phase12Config, run_phase12_campaign
from .baseline_campaign import BaselineConfig, run_baseline_campaign

SwitchGlobeConfig = Phase12Config
run_switchglobe_campaign = run_phase12_campaign

__all__ = [
    "Phase7Config",
    "Phase8Config",
    "Phase11Config",
    "Phase12Config",
    "BaselineConfig",
    "SwitchGlobeConfig",
    "run_phase7_campaign",
    "run_phase8_campaign",
    "run_phase11_campaign",
    "run_phase12_campaign",
    "run_baseline_campaign",
    "run_switchglobe_campaign",
]
