"""Export selected paths as independent, reloadable UI datasets."""
from __future__ import annotations

import csv
import json
import re
import shutil
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path
from itertools import combinations

from TopoTunnel_UI.core import export_statistics as statistics

import numpy as np

from TopoTunnel_UI.core.exporter import export_paths_to_pdb
from TopoTunnel_UI.db.tunnel_db import TunnelDatabase


def _safe_name(value):
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', str(value)).strip(' .')
    return ('dataset_' + name)[:100] if name else 'dataset'


def _resolve_assets(binding, supplied):
    """Prefer the active UI sources, then files belonging to this dataset."""
    assets = dict(supplied)
    roots = []
    for candidate in (Path(binding.path).parent, Path(binding.folder),
                      Path(binding.residue_statistics_path).parent if binding.residue_statistics_path else None,
                      Path(binding.residue_combination_statistics_path).parent if binding.residue_combination_statistics_path else None):
        if candidate and candidate.is_dir() and candidate not in roots:
            roots.append(candidate)
    names = {
        'reference': ('representative_frame.pdb',),
        'residue_stats': ('residue_statistics.csv', 'residue_static.csv'),
        'combo_stats': ('residue_combination_statistics.csv', 'residue_combination_statitics.csv'),
        'distances': ('residue_pair_distances.npy', 'residue_pair_distance.npy',
                      'residue_pari_distance.npy', 'residue_pari_distances.npy',
                      'residue_pair_distances.csv', 'residue_pair_distance.csv',
                      'residue_pari_distance.csv', 'residue_pari_distances.csv',
                      'residue_distance_relations.csv'),
        'presence_stats': ('residue_pair_path_presence_stats.csv', 'residue_pari_path_presence_stats.csv'),
        'md_path_file': ('MD_path.txt',),
    }
    assets.setdefault('residue_stats', binding.residue_statistics_path)
    assets.setdefault('combo_stats', binding.residue_combination_statistics_path)
    for key, filenames in names.items():
        if assets.get(key) and Path(assets[key]).is_file():
            continue
        assets[key] = ''
        for root in roots:
            for filename in filenames:
                candidate = root / filename
                if candidate.is_file():
                    assets[key] = str(candidate)
                    break
            if assets[key]:
                break
    if not assets.get('md_root'):
        assets['md_root'] = binding.md_root_path
    if not assets.get('md_root') and assets.get('md_path_file'):
        path_file = Path(assets['md_path_file'])
        for line in path_file.read_text(encoding='utf-8-sig').splitlines():
            if line.strip():
                candidate = Path(line.strip().strip("\"'"))
                if not candidate.is_absolute():
                    candidate = path_file.parent / candidate
                if candidate.is_file():
                    candidate = candidate.parent
                if candidate.is_dir():
                    assets['md_root'] = str(candidate.resolve())
                break
    return assets


def _subset_database(source, target, path_ids):
    """Use SQLite backup to include committed WAL data, then prune the copy."""
    with closing(sqlite3.connect(target)) as dest:
        # UI queries may leave temporary selection-table transactions open.
        # Read committed source data on a separate connection to avoid waiting
        # indefinitely on a backup of that same transaction.
        reader = sqlite3.connect(Path(source.db_path).resolve().as_uri() + '?mode=ro', uri=True)
        try:
            reader.backup(dest)
        finally:
            reader.close()
        dest.execute('PRAGMA foreign_keys=OFF')
        dest.execute('CREATE TEMP TABLE export_ids(id INTEGER PRIMARY KEY)')
        dest.executemany('INSERT INTO export_ids VALUES (?)', ((pid,) for pid in path_ids))
        actual = dest.execute('SELECT COUNT(*) FROM paths WHERE id IN (SELECT id FROM export_ids)').fetchone()[0]
        if actual != len(path_ids):
            raise ValueError('Some selected paths no longer exist in the source dataset.')
        for table in ('path_points', 'path_render_data', 'residue_paths'):
            dest.execute(f'DELETE FROM {table} WHERE path_id NOT IN (SELECT id FROM export_ids)')
        dest.execute('DELETE FROM paths WHERE id NOT IN (SELECT id FROM export_ids)')
        dest.execute('DELETE FROM residue_positions WHERE residue_id NOT IN (SELECT residue_id FROM residue_paths)')
        dest.execute('UPDATE residue_positions SET path_count=(SELECT COUNT(*) FROM residue_paths r WHERE r.residue_id=residue_positions.residue_id)')
        dest.execute('DELETE FROM sessions')
        lo, hi = dest.execute('SELECT MIN(frame_id), MAX(frame_id) FROM paths').fetchone()
        metadata = {'path_count': actual, 'frame_min': lo, 'frame_max': hi,
                    'residue_count': dest.execute('SELECT COUNT(*) FROM residue_positions').fetchone()[0],
                    'source_file': 'selected-path export'}
        dest.executemany('INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)',
                         ((key, str(value)) for key, value in metadata.items()))
        dest.commit()
        dest.execute('VACUUM')


def _statistics(db, target, source_stats, reference, combo_source=''):
    names = {}
    if source_stats and Path(source_stats).is_file():
        with open(source_stats, encoding='utf-8-sig', newline='') as handle:
            for row in csv.DictReader(handle):
                names[int(row['Residue_ID'])] = row.get('Residue_name', '')
    if combo_source and Path(combo_source).is_file():
        with open(combo_source, encoding='utf-8-sig', newline='') as handle:
            for row in csv.DictReader(handle):
                ids = statistics.parse_residue_ids(row.get('Residue_IDs', ''))
                for rid, name in zip(ids, str(row.get('Residue_names', '')).split(';')):
                    if not names.get(rid):
                        names[rid] = name
    if reference and Path(reference).is_file():
        for line in Path(reference).read_text(encoding='utf-8', errors='replace').splitlines():
            if line.startswith(('ATOM  ', 'HETATM')):
                try:
                    names.setdefault(int(line[22:26]), line[17:20].strip())
                except ValueError:
                    pass
    residues, combo_stats, presence = {}, {}, {}
    for pid in db.get_path_ids():
        data = db.get_path_for_export(pid)
        path_residues, path_combinations = {}, set()
        for point in data['points']:
            members = {int(point.get(f'res_{i}') or 0) for i in range(1, 5)} - {0}
            point_residues = {rid: names.get(rid, '') for rid in members if rid > 0}
            radius = point.get('radius') or 0.0
            statistics.update_point_residue_stats(residues, point_residues, radius)
            statistics.update_point_combination_stats(combo_stats, point_residues, radius, path_combinations)
            path_residues.update(point_residues)
        statistics.update_path_residue_stats(residues, path_residues)
        statistics.update_path_combination_stats(combo_stats, path_combinations)
        presence.setdefault(str(data['meta']['frame_id']), set()).update(combinations(sorted(path_residues), 2))
    statistics.write_residue_statistics(str(target / 'residue_statistics.csv'), residues)
    combo_file = target / 'residue_combination_statistics.csv'
    statistics.write_residue_combination_statistics(str(combo_file), combo_stats)
    combos, residue_ids = statistics.read_distance_relation_combinations(str(combo_file))
    return combos, residue_ids, presence


def _copy_presence_rows(source, target, combos):
    pairs = {pair for combo in combos for pair in combo['pair_ids']}
    with open(source, encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if not {'Residue_ID_1', 'Residue_ID_2'}.issubset(fields):
            raise ValueError('Invalid source path-presence statistics CSV.')
        with target.open('w', encoding='utf-8', newline='') as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            for row in reader:
                key = tuple(sorted((int(row['Residue_ID_1']), int(row['Residue_ID_2']))))
                if key in pairs:
                    writer.writerow(row)


def _read_distance_rows(source, combo_source, needed):
    """Normalize supported stored NPY/CSV distances into cumulative rows."""
    result = {}
    frame_file = source.with_name('residue_distance_frames.json')
    if source.suffix.lower() == '.npy':
        if not combo_source.is_file():
            raise ValueError('The distance NPY requires its source combination CSV.')
        original = np.load(source, mmap_mode='r', allow_pickle=False)
        try:
            with combo_source.open(encoding='utf-8-sig', newline='') as handle:
                rows = list(csv.DictReader(handle))
            if original.ndim != 2 or len(rows) != original.shape[0]:
                raise ValueError('Source distance NPY and combination CSV have inconsistent dimensions.')
            frame_ids = list(range(1, original.shape[1] + 1))
            seen = set()
            for index, row in enumerate(rows):
                key = tuple(sorted(statistics.parse_residue_ids(row['Residue_IDs'])))
                if key in seen:
                    raise ValueError('Duplicate residue combinations in source distance index.')
                seen.add(key)
                if key in needed:
                    result[key] = np.array(original[index], copy=True)
        finally:
            original._mmap.close()
    elif source.suffix.lower() == '.csv':
        with source.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            columns = [column for column in (reader.fieldnames or []) if column.lower().startswith('frame_')]
            if not columns:
                raise ValueError('Distance CSV has no Frame_<id> columns.')
            frame_ids = [int(column.split('_', 1)[1]) for column in columns]
            combo_format = 'Residue_IDs' in (reader.fieldnames or [])
            if not combo_format and not {'Residue_ID_1', 'Residue_ID_2'}.issubset(reader.fieldnames or []):
                raise ValueError('Unsupported distance CSV: expected residue pair or combination IDs.')
            for row in reader:
                key = tuple(sorted(statistics.parse_residue_ids(row['Residue_IDs']))) if combo_format else tuple(sorted((int(row['Residue_ID_1']), int(row['Residue_ID_2']))))
                if key not in needed:
                    continue
                if key in result:
                    raise ValueError('Duplicate selected residue combinations in distance CSV.')
                cells = [str(row.get(column) or '').strip() for column in columns]
                paired = any(';' in cell for cell in cells)
                cumulative = 'cumulative' in str(row.get('Value_definition', '')).lower()
                if combo_format and not paired and not cumulative:
                    raise ValueError('Combination CSV must specify cumulative values or distance;cumulative cells.')
                values = np.asarray([float(cell.split(';')[-1]) if cell else np.nan for cell in cells], dtype=np.float64)
                if not paired and not cumulative:
                    missing = ~np.isfinite(values)
                    values = np.cumsum(np.where(missing, 0.0, values))
                    values[missing] = np.nan
                result[key] = values
    else:
        raise ValueError('Unsupported distance file format.')
    if frame_file.is_file():
        explicit = json.loads(frame_file.read_text(encoding='utf-8'))
        if len(explicit) != len(frame_ids):
            raise ValueError('Distance frame index has the wrong number of columns.')
        frame_ids = [int(frame) for frame in explicit]
    if frame_ids != sorted(set(frame_ids)):
        raise ValueError('Distance frame IDs must be unique and increasing.')
    return result, frame_ids


def _distances(target, binding, assets, combos, presence, warnings):
    """Extract existing distance rows; never calculate distances from structures."""
    pairs = sorted({pair for combo in combos for pair in combo['pair_ids']})
    pair_items = [(pair, i) for i, pair in enumerate(pairs)]
    md_root = Path(assets['md_root']) if assets.get('md_root') else None
    frames = sorted(md_root.glob('*.pdb')) if md_root and md_root.is_dir() else []
    if frames:
        md_target = target / 'md_frames'
        md_target.mkdir()
        for frame in frames:
            shutil.copy2(frame, md_target / frame.name)
        (target / 'MD_path.txt').write_text('md_frames', encoding='utf-8')
    else:
        warnings.append('MD structures unavailable; frame-by-frame protein viewing is unavailable.')
    source = Path(assets.get('distances') or '')
    combo_source = Path(assets.get('combo_stats') or '')
    if not source.is_file():
        warnings.append('No existing distance file is available; distance-dependent files cannot be extracted.')
        return
    needed = {tuple(combo['residue_ids']) for combo in combos} | set(pairs)
    prefixes, frame_ids = _read_distance_rows(source, combo_source, needed)
    keys = [tuple(combo['residue_ids']) for combo in combos]
    for combo in combos:
        key = tuple(combo['residue_ids'])
        if key not in prefixes and all(pair in prefixes for pair in combo['pair_ids']):
            prefixes[key] = np.sum([prefixes[pair] for pair in combo['pair_ids']], axis=0)
    if any(key not in prefixes for key in keys):
        warnings.append('Existing distances do not cover every exported residue combination; the distance matrix was not assembled.')
        return
    matrix = np.asarray([prefixes[key] for key in keys]).reshape(len(keys), len(frame_ids))
    np.save(target / 'residue_pair_distances.npy', matrix, allow_pickle=False)
    (target / 'residue_distance_frames.json').write_text(json.dumps(frame_ids), encoding='utf-8')
    if any(pair not in prefixes for pair in pairs):
        warnings.append('Combination distances were exported, but individual pair distances are missing for presence statistics.')
        return
    values = np.asarray([np.diff(prefixes[pair], prepend=0.0) for pair in pairs]).reshape(len(pairs), len(frame_ids))
    unmatched_frames = set(presence) - {str(frame) for frame in frame_ids}
    if unmatched_frames:
        warnings.append('Some selected-path frame IDs are absent from the structural distance data; presence statistics cover available frames only.')
    statistics.write_pair_path_presence_statistics(
        str(target / 'residue_pair_path_presence_stats.csv'), pair_items, values, frame_ids,
        presence_accumulator=presence)


def export_ui_datasets(db, path_ids, category_name, export_dir, *, split_frame_paths=False, assets=None):
    """Publish a new bundle only after all selected datasets export successfully."""

    grouped = {}
    for pid in sorted({int(pid) for pid in path_ids}):
        binding, local_id = db._decode_path_id(pid)
        if binding is None:
            raise ValueError(f'Unknown selected path: {pid}')
        grouped.setdefault(binding.key, (binding, []))[1].append(local_id)
    if not grouped:
        raise ValueError('No paths selected for export.')
    parent = Path(export_dir).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    base = _safe_name(category_name) + '_export'
    destination = parent / base
    suffix = 2
    while destination.exists():
        destination = parent / f'{base}_{suffix}'
        suffix += 1
    report = {'directory': str(destination), 'datasets': [], 'warnings': [], 'pdb_count': 0}
    with tempfile.TemporaryDirectory(prefix='.topotunnel-export-', dir=parent) as temporary:
        stage = Path(temporary) / 'bundle'
        stage.mkdir()
        for key, (binding, ids) in grouped.items():
            target = stage / f'{_safe_name(binding.prefix)}_{binding.slot}'
            target.mkdir()
            supplied = _resolve_assets(binding, (assets or {}).get(key, {}))
            warnings = []
            _subset_database(binding.db, target / 'preprocessed_paths.db', ids)
            reference = supplied.get('reference', '')
            if reference and Path(reference).is_file():
                shutil.copy2(reference, target / 'representative_frame.pdb')
            elif supplied.get('reference_text'):
                (target / 'representative_frame.pdb').write_text(supplied['reference_text'], encoding='utf-8')
                reference = str(target / 'representative_frame.pdb')
            else:
                warnings.append('Source representative_frame.pdb is missing; reference protein viewing is unavailable.')
            for name, path in supplied.get('shared_view_files', {}).items():
                if name not in ('loaded_protein.pdb', 'loaded_session.pse') or not Path(path).is_file():
                    continue
                context = stage / 'view_context'
                context.mkdir(exist_ok=True)
                shutil.copy2(path, context / name)
            subset = TunnelDatabase(str(target / 'preprocessed_paths.db'))
            subset.connect()
            try:
                pdbs = export_paths_to_pdb(subset, ids, _safe_name(category_name), str(target / 'pdb'), split_frame_paths)
                combos, residue_ids, presence = _statistics(
                    subset, target, supplied.get('residue_stats', ''), reference, supplied.get('combo_stats', ''))
                _distances(target, binding, supplied, combos, presence, warnings)
                # A full-dataset selection may reuse original presence results
                # even if the per-frame distances were not supplied. Subsets
                # cannot reuse aggregate present/absent counts as their own.
                presence_source = supplied.get('presence_stats')
                presence_target = target / 'residue_pair_path_presence_stats.csv'
                if not presence_target.exists() and presence_source and set(ids) == set(binding.db.get_path_ids()):
                    _copy_presence_rows(presence_source, presence_target, combos)
            finally:
                subset.close()
            report['pdb_count'] += len(pdbs)
            files = sorted(str(path.relative_to(target)).replace('\\', '/') for path in target.rglob('*') if path.is_file())
            required = ['preprocessed_paths.db', 'residue_statistics.csv', 'residue_combination_statistics.csv',
                        'representative_frame.pdb', 'residue_pair_distances.npy', 'residue_pair_path_presence_stats.csv', 'MD_path.txt']
            item = {'folder': target.name, 'source_prefix': binding.prefix, 'path_ids': ids, 'warnings': warnings,
                    'files': files, 'missing_files': [name for name in required if name not in files],
                    'sources': {name: supplied.get(name, '') for name in ('reference', 'residue_stats', 'combo_stats', 'distances', 'md_root')}}
            report['datasets'].append(item)
            report['warnings'].extend(f'{binding.prefix}: {warning}' for warning in warnings)
        report['statistics_scope'] = 'Selected stored path points; distance context includes all available source frames. Sessions are not exported.'
        (stage / 'export_manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        stage.rename(destination)
    return report
