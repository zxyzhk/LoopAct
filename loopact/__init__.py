"""LoopAct: Closed-Loop Action Refinement for Vision-Language-Action Policies."""
from importlib import import_module as _import_module
__version__ = "0.1.0"
_EXPORTS = {'LoopAct': ('policy', 'LoopAct'), 'LoopActConfig': ('policy', 'LoopActConfig'), 'ActionProposalPlanner': ('action_proposal_planner', 'ActionProposalPlanner'), 'ActionProposalPlannerConfig': ('action_proposal_planner', 'ActionProposalPlannerConfig'), 'ActionContextEncoder': ('action_proposal_planner', 'ActionContextEncoder'), 'VisualRefinementNetwork': ('visual_refinement_network', 'VisualRefinementNetwork'), 'CrossModalFusionBlock': ('visual_refinement_network', 'CrossModalFusionBlock'), 'VisualObservationEncoder': ('visual_observation_encoder', 'VisualObservationEncoder')}
__all__ = list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attribute = _EXPORTS[name]
    return getattr(_import_module("loopact.model." + module), attribute)
