from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile
from unittest import mock

from lineage import object_hash, sha256_file
from scripts.pack_kaggle import pack
from scripts.run_training import preflight


class KaggleWorkflowTests(unittest.TestCase):
    def fixture(self, root):
        for directory in ('scripts','data/eval','data/seed','data/prepared/pretrain','data/prepared/sft_train','data/prepared/sft_val'):
            (root/directory).mkdir(parents=True,exist_ok=True)
        for name in ('config.py','scripts/runner.py','bpe.json','requirements.txt','hf_datasets.json','sources.json',
                     'data/eval/probes.jsonl','data/eval/heldout.txt','data/seed/conversations.jsonl','parent.pt'):
            (root/name).write_text('{}')
        prepared=root/'data/prepared'
        (prepared/'pretrain/vi.txt').write_text('training text')
        (prepared/'pretrain/validation.txt').write_text('validation text')
        (prepared/'pretrain_sources.jsonl').write_text('')
        mixture={'domains':{'vi':'D:/old/vi.txt'}}
        report={'status':'ready','mixture_hash':object_hash(mixture),'pretrain':{'vi':{'train_sha256':sha256_file(str(prepared/'pretrain/vi.txt'))}},
                'validation_sha256':sha256_file(str(prepared/'pretrain/validation.txt')),'sft':{}}
        (prepared/'mixture.json').write_text(json.dumps(mixture));(prepared/'manifest.json').write_text(json.dumps(report))
        (root/'cache').mkdir();(root/'cache/private.txt').write_text('do not upload')
        return prepared

    def test_pack_only_required_files_and_no_parent_copy(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);prepared=self.fixture(root)
            report=pack(root/'upload',prepared,root/'parent.pt',root=root)
            self.assertFalse((root/'upload/parent.pt').exists())
            with zipfile.ZipFile(root/'upload/jin-code.zip') as z:
                self.assertIn('config.py',z.namelist());self.assertFalse(any(n.startswith('cache/') for n in z.namelist()))
            self.assertEqual(report['parent']['sha256'],sha256_file(str(root/'parent.pt')))
            with self.assertRaisesRegex(ValueError,'already exists'):pack(root/'upload',prepared,root/'parent.pt',root=root)

    def test_tampered_prepared_fails_before_output_created(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);prepared=self.fixture(root)
            (prepared/'pretrain/vi.txt').write_text('tampered')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):pack(root/'upload',prepared,root/'parent.pt',root=root)
            self.assertFalse((root/'upload').exists())

    def test_launcher_accepts_explicit_sft_validation(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'bpe.json').write_text('{}')
            spec={'schema_version':1,'inputs':{'tokenizer':{'path':'bpe.json','sha256':sha256_file(str(root/'bpe.json'))}},
                  'training':{'mode':'instruct','sft-val-dir':'data/prepared/sft_val'},'output_checkpoint':'new.pt'}
            command,_,_=preflight(spec,root=root)
            self.assertIn('--sft-val-dir',command)

    def setup_source(self, inp, work):
        notebook=json.loads((Path(__file__).resolve().parents[1]/'docs/JIN_KAGGLE_V4.ipynb').read_text(encoding='utf-8'))
        source=''.join(notebook['cells'][1]['source'])
        source=source.replace("INPUT = Path('/kaggle/input')","INPUT = Path(%r)"%str(inp))
        source=source.replace("ROOT = Path('/kaggle/working/jin')","ROOT = Path(%r)"%str(work))
        self.assertIn("INPUT = Path(%r)"%str(inp),source);self.assertIn("ROOT = Path(%r)"%str(work),source)
        return source

    def run_setup(self, source, work):
        namespace={}
        with mock.patch('storage.configure_runtime'),mock.patch('os.chdir'),mock.patch.dict(os.environ):
            exec(compile(source,'kaggle-setup','exec'),namespace)
        sys.path.remove(str(work))
        return namespace

    def upload_bundle(self, dataset, checkpoint_dir):
        entries={'jin-code.zip':{'config.py':'verified code','scripts/runner.py':'runner'},
                 'jin-data.zip':{'data/prepared_v4/pretrain/vi.txt':'training text'}}
        archives=[]
        for name,files in entries.items():
            target=dataset/name
            with zipfile.ZipFile(target,'w') as z:
                for entry,text in files.items():z.writestr(entry,text)
            archives.append({'filename':name,'bytes':target.stat().st_size,'sha256':sha256_file(str(target)),
                             'files':[{'path':entry,'bytes':len(text),'sha256':hashlib.sha256(text.encode()).hexdigest()} for entry,text in files.items()]})
        parent=checkpoint_dir/'jingpt_chat.pt';parent.write_text('parent weights')
        bundle={'schema_version':1,'prepared_relative':'data/prepared_v4','archives':archives,
                'parent':{'filename':'jingpt_chat.pt','bytes':parent.stat().st_size,'sha256':sha256_file(str(parent))}}
        return bundle,entries

    def test_notebook_setup_locates_nested_dataset_and_pinned_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);inp=root/'input';work=root/'work'
            dataset=inp/'datasets/annguyen0410/jin-v4-training';dataset.mkdir(parents=True)
            checkpoint_dir=inp/'jin-ckpt';checkpoint_dir.mkdir()
            decoy=inp/'old-jin-ckpt';decoy.mkdir()
            bundle,_=self.upload_bundle(dataset,checkpoint_dir)
            (decoy/'jingpt_chat.pt').write_text('stale parent')
            (dataset/'upload_manifest.json').write_text(json.dumps(bundle))
            namespace=self.run_setup(self.setup_source(inp,work),work)
            self.assertEqual(namespace['DATA_INPUT'],dataset)
            self.assertEqual(namespace['CHECKPOINT_INPUT'],checkpoint_dir/'jingpt_chat.pt')
            self.assertEqual((work/'config.py').read_text(),'verified code')
            self.assertEqual((work/'scripts/runner.py').read_text(),'runner')
            self.assertEqual((work/'data/prepared_v4/pretrain/vi.txt').read_text(),'training text')

    def test_notebook_setup_handles_kaggle_auto_unzip_flat(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);inp=root/'input';work=root/'work'
            dataset=inp/'jin-v4-training';dataset.mkdir(parents=True)
            checkpoint_dir=inp/'jin-ckpt';checkpoint_dir.mkdir()
            bundle,entries=self.upload_bundle(dataset,checkpoint_dir)
            for name,files in entries.items():
                (dataset/name).unlink()
                for entry,text in files.items():
                    target=dataset/entry;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(text)
            (dataset/'upload_manifest.json').write_text(json.dumps(bundle))
            namespace=self.run_setup(self.setup_source(inp,work),work)
            self.assertEqual(namespace['DATA_INPUT'],dataset)
            self.assertEqual((work/'config.py').read_text(),'verified code')
            self.assertEqual((work/'data/prepared_v4/pretrain/vi.txt').read_text(),'training text')

    def test_notebook_setup_handles_kaggle_auto_unzip_subfolders(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);inp=root/'input';work=root/'work'
            dataset=inp/'jin-v4-training';dataset.mkdir(parents=True)
            checkpoint_dir=inp/'jin-ckpt';checkpoint_dir.mkdir()
            bundle,entries=self.upload_bundle(dataset,checkpoint_dir)
            for name,files in entries.items():
                base=dataset/name[:-4]
                (dataset/name).unlink()
                for entry,text in files.items():
                    target=base/entry;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(text)
            (dataset/'upload_manifest.json').write_text(json.dumps(bundle))
            namespace=self.run_setup(self.setup_source(inp,work),work)
            self.assertEqual((work/'scripts/runner.py').read_text(),'runner')
            self.assertEqual((work/'data/prepared_v4/pretrain/vi.txt').read_text(),'training text')
    def test_notebook_pretrain_detects_parent_optimizer(self):
        nb=json.loads((Path(__file__).resolve().parents[1]/'docs/JIN_KAGGLE_V4.ipynb').read_text(encoding='utf-8'))
        cell=''.join(nb['cells'][3]['source'])
        from bpe_tokenizer import BPETokenizer
        for groups,expected in (([{'betas':(.9,.95)}],'adamw'),
                                ([{'betas':(.9,.95),'mode':'muon'},{'betas':(.9,.95),'mode':'adamw'}],'muon')):
            with self.subTest(optimizer=expected),tempfile.TemporaryDirectory() as td:
                root=Path(td);prepared=self.fixture(root)
                (root/'bpe.json').write_text(json.dumps(BPETokenizer().to_state_dict()))
                namespace={'ROOT':root,'json':json,'bundle':{'prepared_relative':'data/prepared'},'MODE':'pretrain','CHECKPOINT_INPUT':root/'parent.pt'}
                with mock.patch('data_pipeline.verify_prepared',return_value={'status':'verified'}), \
                     mock.patch('utils.load_checkpoint_file',return_value={'tokenizer_state':BPETokenizer().to_state_dict(),
                            'optimizer_state_dict':{'param_groups':groups}}):
                    exec(compile(cell,'optimizer-detection','exec'),namespace)
                self.assertEqual(namespace['PARENT_OPTIMIZER'],expected)
    def test_notebook_long_run_guard_requires_explicit_allow(self):
        nb=json.loads((Path(__file__).resolve().parents[1]/'docs/JIN_KAGGLE_V4.ipynb').read_text(encoding='utf-8'))
        cell=''.join(nb['cells'][5]['source'])
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            namespace={'RUN_LONG':True,'STEPS':60000,'ALLOW_LONG':False,'ROOT':root,'OUTPUT':root/'out.pt',
                       'recipe':lambda *a,**k:{'steps':a[2]},'atomic_json':lambda *a,**k:None,'sys':sys,
                       'subprocess':mock.Mock()}
            with self.assertRaisesRegex(AssertionError,'vượt trần'):
                exec(compile(cell,'long-run-guard','exec'),dict(namespace))
            namespace['ALLOW_LONG']=True
            exec(compile(cell,'long-run-guard','exec'),dict(namespace))
            namespace['subprocess'].run.assert_called()

    def test_notebook_cells_compile_and_setup_relocates_verified_mixture(self):
        notebook_path=Path(__file__).resolve().parents[1]/'docs/JIN_KAGGLE_V4.ipynb'
        nb=json.loads(notebook_path.read_text(encoding='utf-8'))
        for i,cell in enumerate(nb['cells']):
            if cell['cell_type']=='code':compile(''.join(cell['source']),f'kaggle-cell-{i}','exec')
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);prepared=self.fixture(root)
            from bpe_tokenizer import BPETokenizer
            (root/'bpe.json').write_text(json.dumps(BPETokenizer().to_state_dict()))
            namespace={'ROOT':root,'json':json,'bundle':{'prepared_relative':'data/prepared'},'MODE':'instruct','CHECKPOINT_INPUT':root/'parent.pt'}
            cell=''.join(nb['cells'][3]['source'])
            with mock.patch('data_pipeline.verify_prepared',return_value={'status':'verified'}), \
                 mock.patch('utils.load_checkpoint_file',return_value={'tokenizer_state':BPETokenizer().to_state_dict()}):
                exec(compile(cell,'relocation-cell','exec'),namespace)
            m=json.loads((prepared/'mixture.json').read_text());r=json.loads((prepared/'manifest.json').read_text())
            self.assertEqual(m['domains']['vi'],str(prepared/'pretrain/vi.txt'))
            self.assertEqual(r['mixture_hash'],object_hash(m))

if __name__=='__main__':unittest.main()
