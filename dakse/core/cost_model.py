"""
Cost Model for DA-KSE.
Estimates the expected marginal execution latency Delta T(a_t) of exploration actions:
Delta T(a_t) = c_op(a_t) + c_level(l(k_i)) + c_tok * |x_i|
"""

from typing import Optional
from dakse.config import PlatformCostProfile, get_platform_profile, RPI5_PROFILE
from dakse.core.knowledge_space import KnowledgeUnit


class CostModel:
    """Profiles and estimates operation latency Delta T(a_t) in milliseconds."""

    def __init__(self, platform_profile: Optional[PlatformCostProfile] = None):
        self.profile = platform_profile or RPI5_PROFILE

    def set_platform(self, platform_name: str) -> None:
        self.profile = get_platform_profile(platform_name)

    def estimate_action_cost_ms(self, action_name: str, target_unit: Optional[KnowledgeUnit]) -> float:
        """
        Estimates expected execution time Delta T(a_t) in milliseconds.
        Returns:
            Estimated latency in ms.
        """
        if action_name == "Stop":
            return 0.0

        c_op = self.profile.op_costs_ms.get(action_name, 10.0)

        if target_unit is None:
            return c_op

        # Granularity level overhead: c_level * l
        c_level = self.profile.level_overhead_ms * target_unit.level

        # Context insertion / token processing cost: c_tok * |x_i|
        c_tok = self.profile.token_cost_ms * target_unit.token_count

        return c_op + c_level + c_tok

    def estimate_action_cost_seconds(self, action_name: str, target_unit: Optional[KnowledgeUnit]) -> float:
        """Returns estimated latency in seconds."""
        return self.estimate_action_cost_ms(action_name, target_unit) / 1000.0
