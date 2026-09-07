import json

from biohub.paths import PROJECT_ROOT

PRODUCTION_NOTEBOOK = (
    PROJECT_ROOT / 'tests' / 'fixtures' / 'legacy_scorer' / 'production_notebook.ipynb'
)
PRODUCTION_NOTEBOOK_SHA256 = '0c141c83ce67fb24a4c30fcc65b06856331f2aa17119bac8f36502fb337f7ec2'


def notebook_cell_source(index: int) -> str:
    payload = json.loads(PRODUCTION_NOTEBOOK.read_text())
    source = ''.join(payload['cells'][index]['source'])
    if source.startswith('%%writefile'):
        source = source.split('\n', 1)[1]
    return source
