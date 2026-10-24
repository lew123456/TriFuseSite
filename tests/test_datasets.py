import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from trifusesite.cli.validate_datasets import validate
from trifusesite.cli.prepare_datasets import load_catalog, load_structure_index, resolve_records
from trifusesite.cli.fetch_structures import fetch, is_structure

ROOT=Path(__file__).resolve().parents[1]

class BundledDatasetTests(unittest.TestCase):
    def test_real_partition_counts_and_content(self):
        result=validate(checksums=False)
        self.assertEqual(result['status'],'pass')
        self.assertEqual(len(result['partitions']),14)
        self.assertEqual(sum(r['proteins'] for r in result['partitions']),4369)

    def test_pair_mapping_is_consumable_jsonl(self):
        rows=[json.loads(s) for s in (ROOT/'dataset/PPI/pairs.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertEqual(len(rows),25)
        self.assertEqual(len({r['bound_id'] for r in rows}),25)
        self.assertEqual(len({r['unbound_id'] for r in rows}),25)

    def test_uninstalled_coordinates_are_not_auto_masked(self):
        spec=next(s for s in load_catalog() if s['task']=='smpi' and s['split']=='test')
        original,_=resolve_records(spec)
        with tempfile.TemporaryDirectory() as tmp:
            index={('smpi',r['id']):{'path':str(Path(tmp)/(r['id']+'.pdb'))} for r in original}
            rows,missing=resolve_records(spec,structure_index=index)
        self.assertEqual(len(rows),348)
        self.assertEqual(len(missing),348)
        self.assertTrue(all(r['structure_path'] is not None for r in rows))

    def test_missing_structure_requires_explicit_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'structures.jsonl'
            path.write_text(json.dumps({'task':'smpi','id':'synthetic','path':None})+'\n',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'requires a recorded reason'):load_structure_index(path)
            path.write_text(json.dumps({'task':'smpi','id':'synthetic','path':None,'reason':'synthetic test fixture'})+'\n',encoding='utf-8')
            self.assertIsNone(load_structure_index(path)[('smpi','synthetic')]['path'])

    def test_structure_index_preserves_records_and_labels(self):
        spec=next(s for s in load_catalog() if s['task']=='smpi' and s['split']=='test')
        baseline,_=resolve_records(spec)
        pid=baseline[0]['id']
        with tempfile.TemporaryDirectory() as tmp:
            index={('smpi',r['id']):{'path':str(Path(tmp)/(r['id']+'.pdb'))} for r in baseline}
            index[('smpi',pid)]={'path':None,'reason':'synthetic override for parser test only'}
            rows,missing=resolve_records(spec,structure_index=index)
        self.assertEqual([r['labels'] for r in rows],[r['labels'] for r in baseline])
        self.assertEqual(len(missing),347)
        self.assertIsNone(rows[0]['structure_path'])

    def test_fetch_checks_payload_without_network(self):
        self.assertFalse(is_structure(b'<html>not found</html>'*20))
        payload=b'HEADER synthetic fixture\nATOM  '+b' '*120
        self.assertTrue(is_structure(payload))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'example.pdb').write_bytes(payload)
            with patch('urllib.request.urlopen') as network:
                result=fetch({'path':'example.pdb','url':'https://example.invalid/example.pdb','source':'PDB'},root)
                self.assertEqual(result['status'],'existing');network.assert_not_called()
            with self.assertRaisesRegex(ValueError,'escapes repository'):
                fetch({'path':'../escape.pdb','url':'https://example.invalid','source':'PDB'},root)

if __name__=='__main__':unittest.main()
