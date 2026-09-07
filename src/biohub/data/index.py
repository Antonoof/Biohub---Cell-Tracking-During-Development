import json
from pathlib import Path
from typing import Any

from biohub.data.movies import embryo_of


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_zarr_shape(zarr_dir: Path) -> tuple[int, ...] | None:
    array_meta = zarr_dir / '0' / 'zarr.json'
    if not array_meta.is_file():
        return None
    payload = _read_json(array_meta)
    shape = payload.get('shape')
    return tuple(int(v) for v in shape) if shape else None


def read_zarr_quantiles(zarr_dir: Path) -> dict[str, float]:
    group_meta = zarr_dir / 'zarr.json'
    if not group_meta.is_file():
        return {}
    payload = _read_json(group_meta)
    quantiles = (payload.get('attributes') or {}).get('image_statistics', {}).get('quantiles') or {}
    out: dict[str, float] = {}
    for key, value in quantiles.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def read_geff_metadata(geff_dir: Path) -> dict[str, Any]:
    meta_path = geff_dir / 'zarr.json'
    if not meta_path.is_file():
        return {}
    payload = _read_json(meta_path)
    return (payload.get('attributes') or {}).get('geff') or {}


def index_split_dir(
    split_dir: Path, *, role_by_id: dict[str, str] | None = None
) -> list[dict[str, Any]]:
    split_dir = Path(split_dir)
    if not split_dir.is_dir():
        raise FileNotFoundError(f'Data directory not found: {split_dir}')
    zarrs = {path.name.removesuffix('.zarr'): path for path in split_dir.glob('*.zarr')}
    geffs = {path.name.removesuffix('.geff'): path for path in split_dir.glob('*.geff')}
    movie_ids = sorted(set(zarrs) | set(geffs))
    rows: list[dict[str, Any]] = []
    for movie_id in movie_ids:
        zarr_dir = zarrs.get(movie_id)
        geff_dir = geffs.get(movie_id)
        geff_meta = read_geff_metadata(geff_dir) if geff_dir else {}
        extra = geff_meta.get('extra') or {}
        axes = {axis['name']: axis for axis in geff_meta.get('axes') or []}
        shape = read_zarr_shape(zarr_dir) if zarr_dir is not None else None
        rows.append(
            {
                'movie_id': movie_id,
                'embryo': embryo_of(movie_id),
                'source_group': embryo_of(movie_id),
                'split_role': (role_by_id or {}).get(movie_id, 'unknown'),
                'estimated_number_of_nodes': extra.get('estimated_number_of_nodes'),
                'image_shape_tzyx': list(shape) if shape is not None else None,
                't_max': (axes.get('t') or {}).get('max'),
                'has_zarr': zarr_dir is not None,
                'has_geff': geff_dir is not None,
                'quantiles': read_zarr_quantiles(zarr_dir) if zarr_dir else {},
            }
        )
    return rows
