import json

from biohub.data.index import index_split_dir
from biohub.data.movies import load_movie_catalog, panel_movie_ids


def test_catalog_covers_199_movies_and_roles() -> None:
    catalog = load_movie_catalog()
    assert catalog['n_movies'] == 199
    roles = {}
    for row in catalog['movies']:
        roles[row['split_role']] = roles.get(row['split_role'], 0) + 1
    assert roles == {'train175': 175, 'held20': 20, 'practice4': 4}


def test_smoke_panel_is_practice_four() -> None:
    assert panel_movie_ids('smoke') == [
        '44b6_0113de3b',
        '44b6_0b24845f',
        '6bba_05b6850b',
        '6bba_05db0fb1',
    ]


def test_regression_panel_excludes_held20() -> None:
    catalog = {row['movie_id']: row for row in load_movie_catalog()['movies']}
    for movie_id in panel_movie_ids('regression'):
        assert catalog[movie_id]['split_role'] in {'practice4', 'train175'}


def test_index_tmp_dataset(tmp_path) -> None:
    movie = tmp_path / '44b6_deadbeef.zarr'
    (movie / '0').mkdir(parents=True)
    (movie / 'zarr.json').write_text(
        json.dumps({'attributes': {'image_statistics': {'quantiles': {'0.001': 1.0}}}})
    )
    (movie / '0' / 'zarr.json').write_text(json.dumps({'shape': [8, 4, 16, 16]}))
    geff = tmp_path / '44b6_deadbeef.geff'
    geff.mkdir()
    (geff / 'zarr.json').write_text(
        json.dumps(
            {
                'attributes': {
                    'geff': {
                        'axes': [{'name': 't', 'max': 7}],
                        'extra': {'estimated_number_of_nodes': 12},
                    }
                }
            }
        )
    )
    rows = index_split_dir(tmp_path, role_by_id={'44b6_deadbeef': 'practice4'})
    assert rows[0]['movie_id'] == '44b6_deadbeef'
    assert rows[0]['embryo'] == '44b6'
    assert rows[0]['estimated_number_of_nodes'] == 12
    assert rows[0]['image_shape_tzyx'] == [8, 4, 16, 16]
