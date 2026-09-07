from pathlib import Path

from biohub.data.submission import load_submission_graphs
from biohub.data.synthetic import continuation_pair


def test_submission_csv_roundtrip_via_polars(tmp_path: Path) -> None:
    pred, _ = continuation_pair()
    path = tmp_path / 'submission.csv'
    lines = [
        'id,dataset,row_type,node_id,t,z,y,x,source_id,target_id',
    ]
    row_id = 0
    for node_id, t, z, y, x in zip(
        pred.node_ids.tolist(),
        pred.t.tolist(),
        pred.z.tolist(),
        pred.y.tolist(),
        pred.x.tolist(),
        strict=True,
    ):
        lines.append(
            f'{row_id},{pred.movie_id},node,{int(node_id)},{int(t)},'
            f'{int(round(z))},{int(round(y))},{int(round(x))},-1,-1'
        )
        row_id += 1
    for src, tgt in zip(pred.source_ids.tolist(), pred.target_ids.tolist(), strict=True):
        lines.append(f'{row_id},{pred.movie_id},edge,-1,-1,-1,-1,-1,{int(src)},{int(tgt)}')
        row_id += 1
    path.write_text('\r\n'.join(lines) + '\r\n')
    loaded = load_submission_graphs(path)[pred.movie_id]
    assert loaded.n_nodes == pred.n_nodes
    assert loaded.n_edges == pred.n_edges
    assert loaded.node_ids.tolist() == [int(v) for v in pred.node_ids.tolist()]
    assert loaded.source_ids.tolist() == [int(v) for v in pred.source_ids.tolist()]
    assert loaded.target_ids.tolist() == [int(v) for v in pred.target_ids.tolist()]
