"""Regression checks for reloadable selected-path exports (unittest)."""
import csv
import json
import tempfile
import unittest
from unittest.mock import patch
import builtins
from pathlib import Path

import numpy as np

from TopoTunnel_UI.db.tunnel_db import TunnelDatabase
from TopoTunnel_UI.db.multi_dataset_db import MultiDatasetDatabase
from TopoTunnel_UI.core.dataset_export import export_ui_datasets


class DatasetExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.md = self.source / 'frames'
        self.md.mkdir()
        for frame, distance in [(10, 3), (20, 5)]:
            atoms = []
            for resid, x in [(1, 0), (2, distance)]:
                atoms.append(f'ATOM  {resid:5d}  CA  ALA A{resid:4d}    {x:8.3f}{0:8.3f}{0:8.3f}  1.00  0.00           C  ')
            (self.md / f'frame_{frame}.pdb').write_text('\n'.join(atoms))
        self.reference = self.source / 'representative_frame.pdb'
        self.reference.write_bytes((self.md / 'frame_10.pdb').read_bytes())
        db = TunnelDatabase(str(self.source / 'paths.db'))
        db.connect()
        db.init_schema()
        for pid, frame, radius in [(1, 10, 2), (2, 20, 9)]:
            db.conn.execute('INSERT INTO paths(id,frame_id,point_count) VALUES (?,?,?)', (pid, frame, 1))
            db.conn.execute('INSERT INTO path_points(path_id,seq,x,y,z,radius,res_1,res_2) VALUES (?,0,0,0,0,?,1,2)', (pid, radius))
            db.conn.execute('INSERT INTO path_render_data(path_id,render_coords) VALUES (?,?)', (pid, np.zeros((1, 3), dtype=np.float32).tobytes()))
            db.conn.executemany('INSERT INTO residue_paths VALUES (?,?)', [(1, pid), (2, pid)])
        db.conn.executemany('INSERT INTO residue_positions VALUES (?,0,0,0,2)', [(1,), (2,)])
        db.conn.commit()
        db.close()
        # Existing distances deliberately differ from the PDB geometry. Export
        # must preserve these values instead of recalculating from coordinates.
        self.combo_csv = self.source / 'residue_combination_statistics.csv'
        self.combo_csv.write_text('Combination_size,Residue_IDs,Residue_names\n2,1;2,ALA;ALA\n', encoding='utf-8')
        np.save(self.source / 'residue_pair_distances.npy', np.array([[30, 80]], dtype=np.float32))
        (self.source / 'residue_distance_frames.json').write_text('[10, 20]', encoding='utf-8')
        self.multi = MultiDatasetDatabase()
        self.key = self.multi.add_dataset(str(self.source / 'paths.db'), prefix='A', md_root_path=str(self.md),
                                         residue_combination_statistics_path=str(self.combo_csv))
        self.assets = {self.key: {'reference': str(self.reference),
                                 'distances': str(self.source / 'residue_pair_distances.npy')}}

    def tearDown(self):
        self.multi.close()
        self.temp.cleanup()

    @staticmethod
    def csv_rows(path):
        with path.open(newline='', encoding='utf-8') as handle:
            return list(csv.DictReader(handle))

    def export(self, **kwargs):
        return export_ui_datasets(self.multi, [10000001], 'test', str(self.root / 'out'), assets=self.assets, **kwargs)

    def test_round_trip_and_statistics(self):
        # Selection queries leave a temporary-table transaction on the UI connection.
        self.multi.primary_db.count_filtered_paths(path_id_filter={1})
        original_import = builtins.__import__
        def forbid_pipeline(name, *args, **kwargs):
            if name.startswith('TopoTunnel_Pipeline'):
                raise AssertionError('Export must not import pipeline modules')
            return original_import(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=forbid_pipeline):
            result = self.export(split_frame_paths=True)
        self.assertFalse(result['warnings'])
        folder = Path(result['directory']) / result['datasets'][0]['folder']
        for name in ['preprocessed_paths.db', 'representative_frame.pdb', 'residue_statistics.csv',
                     'residue_combination_statistics.csv', 'residue_pair_distances.npy',
                     'residue_pair_path_presence_stats.csv', 'MD_path.txt']:
            self.assertTrue((folder / name).is_file(), name)
        loaded = TunnelDatabase(str(folder / 'preprocessed_paths.db'))
        loaded.connect()
        try:
            self.assertEqual(loaded.get_path_ids(), [1])
            self.assertEqual(loaded.get_path_for_export(1)['points'][0]['radius'], 2)
            self.assertEqual(loaded.get_meta('path_count'), '1')
        finally:
            loaded.close()
        self.assertEqual(self.multi.primary_db.path_count(), 2)
        rows = self.csv_rows(folder / 'residue_statistics.csv')
        self.assertTrue(all(row['Affected_path_count'] == '1' and float(row['Affected_point_radius_avg']) == 2 for row in rows))
        np.testing.assert_array_equal(np.load(folder / 'residue_pair_distances.npy'), [[30, 80]])
        self.assertEqual(json.loads((folder / 'residue_distance_frames.json').read_text()), [10, 20])
        rows = self.csv_rows(folder / 'residue_pair_path_presence_stats.csv')
        self.assertEqual(rows[0]['Path_present_frame_count'], '1')
        self.assertEqual(rows[0]['Path_absent_frame_count'], '1')
        self.assertEqual(float(rows[0]['Path_present_distance_mean']), 30)
        self.assertEqual(float(rows[0]['Path_absent_distance_mean']), 50)
        self.assertTrue(list((folder / 'pdb' / 'top100').glob('*.pdb')))
        again = MultiDatasetDatabase()
        key = again.add_dataset(str(folder / 'preprocessed_paths.db'),
            residue_statistics_path=str(folder / 'residue_statistics.csv'),
            residue_combination_statistics_path=str(folder / 'residue_combination_statistics.csv'))
        try:
            second = export_ui_datasets(again, [10000001], 'test', str(self.root / 'out'), assets={key: {
                'distances': str(folder / 'residue_pair_distances.npy'), 'reference': str(self.reference)}})
            self.assertNotEqual(second['directory'], result['directory'])
            sf = Path(second['directory']) / second['datasets'][0]['folder']
            np.testing.assert_array_equal(np.load(sf / 'residue_pair_distances.npy'), [[30, 80]])
        finally:
            again.close()

    def test_atomic_failure(self):
        output = self.root / 'out'
        output.mkdir()
        with self.assertRaises(ValueError):
            export_ui_datasets(self.multi, [10000999], 'bad', str(output))
        self.assertEqual(list(output.iterdir()), [])

    def test_missing_distances_do_not_trigger_structure_processing(self):
        self.assets[self.key].pop('distances')
        (self.source / 'residue_pair_distances.npy').unlink()
        result = self.export()
        folder = Path(result['directory']) / result['datasets'][0]['folder']
        self.assertTrue(result['warnings'])
        self.assertTrue((folder / 'MD_path.txt').is_file())
        self.assertTrue((folder / 'md_frames' / 'frame_10.pdb').is_file())
        self.assertFalse((folder / 'residue_pair_distances.npy').exists())
        self.assertFalse((folder / 'residue_pair_path_presence_stats.csv').exists())

    def test_auto_discovery_and_manifest(self):
        result = export_ui_datasets(self.multi, [10000001], 'discover', str(self.root / 'out'))
        item = result['datasets'][0]
        self.assertEqual(item['missing_files'], [])
        self.assertEqual(item['sources']['distances'], str(self.source / 'residue_pair_distances.npy'))

    def test_pair_csv_and_active_observer_folder(self):
        csv_path = self.source / 'pairs.csv'
        csv_path.write_text('Pair_ID,Residue_ID_1,Residue_ID_2,Frame_10,Frame_20\n1-2,1,2,7,11\n', encoding='utf-8')
        active = self.root / 'active_frames'
        active.mkdir()
        (active / 'frame_10.pdb').write_text('REMARK active observer source', encoding='utf-8')
        self.assets[self.key].update(distances=str(csv_path), md_root=str(active))
        result = self.export()
        folder = Path(result['directory']) / result['datasets'][0]['folder']
        np.testing.assert_array_equal(np.load(folder / 'residue_pair_distances.npy'), [[7,18]])
        self.assertEqual((folder / 'md_frames' / 'frame_10.pdb').read_text(), 'REMARK active observer source')
        self.assertFalse((folder / 'md_frames' / 'frame_20.pdb').exists())
        rows = self.csv_rows(folder / 'residue_pair_path_presence_stats.csv')
        self.assertEqual(float(rows[0]['Path_absent_distance_mean']), 11)

    def test_combination_csv_preserves_prefix_values(self):
        csv_path = self.source / 'combinations.csv'
        csv_path.write_text('Combination_size,Residue_IDs,Residue_names,Value_definition,Frame_10,Frame_20\n2,1;2,ALA;ALA,cumulative_sum,13,31\n', encoding='utf-8')
        self.assets[self.key]['distances'] = str(csv_path)
        result = self.export()
        folder = Path(result['directory']) / result['datasets'][0]['folder']
        np.testing.assert_array_equal(np.load(folder / 'residue_pair_distances.npy'), [[13,31]])

    def test_in_memory_reference_and_shared_view_context(self):
        (self.source / 'representative_frame.pdb').unlink()
        self.assets[self.key].update(reference='', reference_text='REMARK loaded observer structure',
                                    shared_view_files={'loaded_protein.pdb': str(self.md / 'frame_10.pdb')})
        result = self.export()
        folder = Path(result['directory']) / result['datasets'][0]['folder']
        self.assertEqual((folder / 'representative_frame.pdb').read_text(), 'REMARK loaded observer structure')
        self.assertTrue((Path(result['directory']) / 'view_context' / 'loaded_protein.pdb').is_file())

    def test_multiple_datasets_and_missing_source_files(self):
        second_db = self.source / 'other.db'
        shutil_source = self.multi.primary_db
        import sqlite3
        with sqlite3.connect(second_db) as conn:
            shutil_source.conn.backup(conn)
        conn.close()
        key = self.multi.add_dataset(str(second_db), prefix='B')
        result = export_ui_datasets(self.multi, [10000001, 20000001], '../CON', str(self.root / 'out'), assets=self.assets)
        self.assertEqual(len(result['datasets']), 2)
        self.assertTrue(result['warnings'])
        for item in result['datasets']:
            folder = Path(result['directory']) / item['folder']
            loaded = TunnelDatabase(str(folder / 'preprocessed_paths.db'))
            loaded.connect()
            try:
                self.assertEqual(loaded.get_path_ids(), [1])
            finally:
                loaded.close()


if __name__ == '__main__':
    unittest.main()
