from biohub.data.motion import ProposalWindowDataset
from biohub.utils.seed import sample_rng


class _FakeVideo:
    stem = '44b6_x'
    image_shape_raw = (4,)


def test_proposal_window_choice_is_seeded_by_index(monkeypatch) -> None:
    monkeypatch.setattr('biohub.data.motion.window_counts', lambda *args, **kwargs: (1, 1, 1, 1))
    dataset = ProposalWindowDataset(
        videos=[_FakeVideo()],
        max_nodes=8,
        train=True,
        steps_per_epoch=4,
        batch_size=2,
        seed=1337,
    )

    def choose(index: int, epoch: int = 0):
        rng = sample_rng(dataset.seed, epoch, index)
        return dataset._choose(index, rng).t

    first = [choose(index) for index in range(16)]
    second = [choose(index) for index in range(16)]
    assert first == second
    dataset.set_epoch(1)
    later = [choose(index, epoch=1) for index in range(16)]
    assert later == [choose(index, epoch=1) for index in range(16)]
    assert later != first or len(set(first)) == 1
    assert len(set(first)) >= 1
