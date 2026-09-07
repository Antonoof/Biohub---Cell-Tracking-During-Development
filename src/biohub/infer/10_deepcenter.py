from biohub.modules.graph.upgrade import GraphUpgrade


def apply(upgrade: GraphUpgrade):
    device = str(upgrade.deepcenter_device)
    if device.startswith('cuda'):
        return None
    return upgrade.load_deepcenter()
