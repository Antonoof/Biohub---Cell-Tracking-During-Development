MOTION_FEATURES = (
    'base_cost',
    'raw_dist',
    'registered_dist',
    'motion_dist',
    'abs_dz',
    'abs_dy',
    'abs_dx',
    'abs_reg_dz',
    'abs_reg_dy',
    'abs_reg_dx',
    'velocity_z',
    'velocity_y',
    'velocity_x',
    'velocity_mag',
    'shift_z',
    'shift_y',
    'shift_x',
    'shift_mag',
    'density_src',
    'density_tgt',
    'z_boundary_src',
    'z_boundary_tgt',
    'has_predecessor',
)

MOTION_TRAIN_FEATURES = (
    *MOTION_FEATURES[:18],
    'det_src',
    'det_tgt',
    'det_disagree_src',
    'det_disagree_tgt',
    *MOTION_FEATURES[18:],
    'frozen',
)

RUNTIME_DROP = {'det_src', 'det_tgt', 'det_disagree_src', 'det_disagree_tgt', 'frozen'}
