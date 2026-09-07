from biohub.modules.graph.upgrade import GraphUpgrade


def apply(upgrade: GraphUpgrade):
    return upgrade.load_deepcenter()
