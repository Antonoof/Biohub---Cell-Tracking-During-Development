import json
from pathlib import Path
from typing import Any

from biohub.constants import VOXEL_SCALE_ZYX
from biohub.contracts import MovieRecord
from biohub.validation.splits import catalog_path, load_split, panel_movie_ids

__all__ = [
    'embryo_of',
    'load_movie_catalog',
    'load_split',
    'movie_record_from_catalog',
    'panel_movie_ids',
]


def embryo_of(movie_id: str) -> str:
    return movie_id.split('_', 1)[0]


def load_movie_catalog(path: Path | None = None) -> dict[str, Any]:
    catalog_path_resolved = path or catalog_path()
    return json.loads(Path(catalog_path_resolved).read_text())


def movie_record_from_catalog(
    row: dict[str, Any],
    train_dir: Path,
    voxel_scale_um: tuple[float, float, float] = VOXEL_SCALE_ZYX,
) -> MovieRecord:
    movie_id = str(row['movie_id'])
    shape = row.get('image_shape_tzyx')
    zarr_path = train_dir / f'{movie_id}.zarr'
    geff_path = train_dir / f'{movie_id}.geff'
    return MovieRecord(
        movie_id=movie_id,
        embryo=str(row.get('embryo') or embryo_of(movie_id)),
        source_group=str(row.get('source_group') or embryo_of(movie_id)),
        split_role=str(row.get('split_role') or 'unknown'),
        zarr_path=zarr_path if zarr_path.exists() else None,
        geff_path=geff_path if geff_path.exists() else None,
        image_shape_tzyx=tuple(shape) if shape else None,
        estimated_number_of_nodes=(
            float(row['estimated_number_of_nodes'])
            if row.get('estimated_number_of_nodes') is not None
            else None
        ),
        t_max=row.get('t_max'),
        voxel_scale_um=voxel_scale_um,
        quantiles={str(k): float(v) for k, v in (row.get('quantiles') or {}).items()},
    )
