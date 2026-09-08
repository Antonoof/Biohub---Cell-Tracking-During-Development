from pathlib import Path

from biohub.train.tensorboard import log_scalars, open_writer


def test_open_writer_logs_scalars(tmp_path: Path) -> None:
    writer = open_writer(tmp_path)
    log_scalars(
        writer,
        3,
        {'val/edge_f1': 0.5, 'val/competition_metric': 0.2, 'skip': None, 'nan': float('nan')},
    )
    writer.close()
    log_dir = tmp_path / 'tensorboard'
    assert log_dir.is_dir()
    events = list(log_dir.glob('events.out.tfevents.*'))
    assert events
    assert events[0].stat().st_size > 0
