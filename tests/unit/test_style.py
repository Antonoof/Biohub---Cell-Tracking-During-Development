import ast
import tomllib

from biohub.paths import PROJECT_ROOT


def test_package_has_no_future_annotations() -> None:
    root = PROJECT_ROOT / 'src' / 'biohub'
    offenders = []
    for path in root.rglob('*.py'):
        for line in path.read_text().splitlines():
            if line.strip() == 'from __future__ import annotations':
                offenders.append(str(path.relative_to(PROJECT_ROOT)))
                break
    assert offenders == []


def test_src_biohub_data_is_not_gitignored() -> None:
    gitignore = (PROJECT_ROOT / '.gitignore').read_text().splitlines()
    assert 'data/' not in gitignore
    assert '/data/' in gitignore
    assert (PROJECT_ROOT / 'src' / 'biohub' / 'data' / 'graph.py').is_file()


def _scan_body(body: list, path, offenders: list) -> None:
    for stmt in body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            offenders.append(f'{path}:{stmt.lineno}')
            continue
        for attr in ('body', 'orelse', 'finalbody'):
            nested = getattr(stmt, attr, None)
            if nested:
                _scan_body(nested, path, offenders)
        handlers = getattr(stmt, 'handlers', None)
        if handlers:
            for handler in handlers:
                _scan_body(handler.body, path, offenders)


def test_no_inner_imports() -> None:
    root = PROJECT_ROOT / 'src' / 'biohub'
    offenders = []
    for path in root.rglob('*.py'):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                _scan_body(node.body, path.relative_to(PROJECT_ROOT), offenders)
    assert offenders == []


def test_src_has_no_legacy_package_hooks() -> None:
    root = PROJECT_ROOT / 'src' / 'biohub'
    offenders = []
    needles = (
        'biohub.production',
        'biohub_tracking',
        'legacy957',
        'spec_from_file_location',
    )
    for path in root.rglob('*.py'):
        text = path.read_text()
        for needle in needles:
            if needle in text:
                offenders.append(f'{path.relative_to(PROJECT_ROOT)}:{needle}')
    assert offenders == []


def test_src_has_no_mass_noqa() -> None:
    root = PROJECT_ROOT / 'src' / 'biohub'
    offenders = []
    for path in root.rglob('*.py'):
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if '# noqa' in line:
                offenders.append(f'{path.relative_to(PROJECT_ROOT)}:{lineno}')
    assert offenders == []


def test_src_has_no_underscore_modules() -> None:
    root = PROJECT_ROOT / 'src' / 'biohub'
    offenders = [
        str(path.relative_to(PROJECT_ROOT))
        for path in root.rglob('*.py')
        if path.name.startswith('_') and path.name != '__init__.py'
    ]
    assert offenders == []


def test_src_does_not_import_private_names() -> None:
    root = PROJECT_ROOT / 'src' / 'biohub'
    offenders = []
    for path in root.rglob('*.py'):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name.startswith('_') and not alias.name.startswith('__'):
                        offenders.append(
                            f'{path.relative_to(PROJECT_ROOT)}:{node.lineno}:{alias.name}'
                        )
                    if (
                        alias.asname
                        and alias.asname.startswith('_')
                        and not alias.asname.startswith('__')
                    ):
                        offenders.append(
                            f'{path.relative_to(PROJECT_ROOT)}:{node.lineno}:{alias.asname}'
                        )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if (
                        alias.asname
                        and alias.asname.startswith('_')
                        and not alias.asname.startswith('__')
                    ):
                        offenders.append(
                            f'{path.relative_to(PROJECT_ROOT)}:{node.lineno}:{alias.asname}'
                        )
    assert offenders == []


def test_src_and_tests_are_not_tool_excluded() -> None:
    payload = tomllib.loads((PROJECT_ROOT / 'pyproject.toml').read_text())
    ruff_exclude = list(payload.get('tool', {}).get('ruff', {}).get('extend-exclude', []))
    ty_exclude = list(payload.get('tool', {}).get('ty', {}).get('src', {}).get('exclude', []))
    for item in [*ruff_exclude, *ty_exclude]:
        text = str(item).replace('\\', '/')
        assert not text.startswith('src')
        assert text.startswith('tests/fixtures/') or not text.startswith('tests')
