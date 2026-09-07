from biohub.modules.graph.upgrade import GraphUpgrade


def apply(upgrade: GraphUpgrade):
    # CUDA tensors are not fork-safe. Workers load DeepCenter lazily after fork.
    device = str(upgrade.deepcenter_device)
    if device.startswith('cuda'):
        return None
    return upgrade.load_deepcenter()
