"""Exact motion-corrector contract excerpt from the current .953 notebook.

This is reference code, not a standalone replacement for the production module.
"""

MOTION_FEATURES = [
    "base_cost", "raw_dist", "registered_dist", "motion_dist",
    "abs_dz", "abs_dy", "abs_dx", "abs_reg_dz", "abs_reg_dy", "abs_reg_dx",
    "velocity_z", "velocity_y", "velocity_x", "velocity_mag",
    "shift_z", "shift_y", "shift_x", "shift_mag",
    "density_src", "density_tgt", "z_boundary_src", "z_boundary_tgt",
    "has_predecessor",
]


class MotionResidual(torch.nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_features, 64),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.05),
            torch.nn.Linear(64, 32),
            torch.nn.SiLU(),
            torch.nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def bounded_residual(model, features, mean, std, residual_scale):
    x = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
    x = (x - mean) / std
    raw = model(x)
    return residual_scale * torch.tanh(raw / residual_scale)


# Current production values:
TIGHT_UM = 6.0
RELAXED_UM = 10.0
VELOCITY_WEIGHT_AXES_ZYX = np.asarray([0.0, 0.45, 0.47])
FRAME_REGISTRATION = True
REGISTRATION_WEIGHT = 1.0
LEARNED_EDGE_PROBABILITY_BONUS = 1.0
RAW_DISTANCE_WEIGHT = 0.05
MAX_MATCH_COST_UM = 7.5
CORRECTOR_STRENGTH = 1.0


# Per frame transition, production computes:
# predicted = source_pos + VELOCITY_WEIGHT_AXES_ZYX * velocity + REGISTRATION_WEIGHT * shift
# motion = norm(target_pos - predicted)
# base_cost = motion + RAW_DISTANCE_WEIGHT * raw_distance
# value = base_cost - LEARNED_EDGE_PROBABILITY_BONUS * edge_probability
# value = value - CORRECTOR_STRENGTH * bounded_residual(...)
#
# The refusal cap is judged against base_cost, before the learned edge bonus and
# learned residual. Surviving candidates enter the Hungarian assignment.
