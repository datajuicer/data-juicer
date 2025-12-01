import os
import os.path as osp
import shutil
import subprocess
import tempfile
import unittest
import uuid
import yaml
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, TEST_TAG


def run_in_subprocess(cmd):
    """Run command in subprocess and capture all output."""
    try:
        # Create a temporary file for logging
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.log') as log_file:
            # Redirect both stdout and stderr to the log file
            process = subprocess.Popen(
                cmd, 
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,  # Line buffered
                encoding='utf-8',
                errors='ignore',
            )

            # Real-time output handling
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    print(line.rstrip())  # Print to console
                    log_file.write(line)  # Write to log file
                    log_file.flush()      # Ensure it's written immediately

            # Get return code
            return_code = process.wait()
            
            # If process failed, read the entire log
            if return_code != 0:
                log_file.seek(0)
                log_content = log_file.read()
                raise RuntimeError(
                    f"Process failed with return code {return_code}.\n"
                    f"Command: {cmd}\n"
                    f"Log output:\n{log_content}"
                )

    except Exception as e:
        print(f"Error running subprocess: {str(e)}")
        raise


class ProcessDataTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()

        self.tmp_dir = tempfile.TemporaryDirectory().name
        os.makedirs(self.tmp_dir, exist_ok=True)

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.tmp_dir)

    def _test_status_code(self, yaml_file, output_path, text_keys):
        data_path = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__)))),
            'demos', 'data', 'demo-dataset.jsonl')
        yaml_config = {
            'dataset_path': data_path,
            'text_keys': text_keys,
            'np': 2,
            'export_path': output_path,
            'process': [
                {
                    'clean_copyright_mapper': None
                }
            ]
        }

        with open(yaml_file, 'w') as file:
            yaml.dump(yaml_config, file)

        script_path = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__)))), 
                               "tools", "process_data.py")
        status_code = subprocess.call(
            f'python {script_path} --config {yaml_file}', shell=True)

        return status_code

    def test_status_code_0(self):
        tmp_yaml_file = osp.join(self.tmp_dir, 'config_0.yaml')
        tmp_out_path = osp.join(self.tmp_dir, 'output_0.json')
        text_keys = 'text'

        status_code = self._test_status_code(tmp_yaml_file, tmp_out_path, text_keys)

        self.assertEqual(status_code, 0)
        self.assertTrue(osp.exists(tmp_out_path))

    def test_status_code_1(self):
        tmp_yaml_file = osp.join(self.tmp_dir, 'config_1.yaml')
        tmp_out_path = osp.join(self.tmp_dir, 'output_1.json')
        text_keys = 'keys_not_exists'

        status_code = self._test_status_code(tmp_yaml_file, tmp_out_path, text_keys)

        self.assertEqual(status_code, 1)
        self.assertFalse(osp.exists(tmp_out_path))


class ProcessDataRayTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()

        cur_dir = osp.dirname(osp.abspath(__file__))
        self.tmp_dir = osp.join(cur_dir, f'tmp_{uuid.uuid4().hex}')
        self.script_path = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__)))), 
                               "tools", "process_data.py")
        os.makedirs(self.tmp_dir, exist_ok=True)

    def tearDown(self):
        super().tearDown()

        if osp.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

        import ray
        ray.shutdown()

    @TEST_TAG("ray")
    def test_ray_image(self):
        tmp_yaml_file = osp.join(self.tmp_dir, 'config_0.yaml')
        tmp_out_path = osp.join(self.tmp_dir, 'output_0.json')
        text_keys = 'text'

        data_path = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__)))),
                             'demos', 'data', 'demo-dataset-images.jsonl')
        yaml_config = {
            'dataset_path': data_path,
            'executor_type': 'ray',
            'ray_address': 'auto',
            'text_keys': text_keys,
            'image_key': 'images',
            'keep_stats_in_res_ds': True,
            'export_path': tmp_out_path,
            'process': [
                {
                    'image_nsfw_filter': {
                        'hf_nsfw_model': 'Falconsai/nsfw_image_detection',
                        'trust_remote_code': True,
                        'max_score': 0.5,
                        'any_or_all': 'any',
                        'mem_required': '8GB'
                    },
                    'image_aspect_ratio_filter':{
                        'min_ratio': 0.5,
                        'max_ratio': 2.0
                    }
                }
            ]
        }

        with open(tmp_yaml_file, 'w') as file:
            yaml.dump(yaml_config, file)

        print(f"Is the config file present? {os.path.exists(tmp_yaml_file)}")
        run_in_subprocess(f'python {self.script_path} --config {tmp_yaml_file}')

        self.assertTrue(osp.exists(tmp_out_path))

        from datasets import load_dataset
        jsonl_files = [os.path.join(tmp_out_path, f) \
                       for f in os.listdir(tmp_out_path) \
                        if f.endswith('.json')]
        dataset = load_dataset(
            'json',
            data_files={'jsonl': jsonl_files})

        self.assertEqual(len(dataset['jsonl']), 3)
        for item in dataset['jsonl']:
            self.assertIn('aspect_ratios', item['__dj__stats__'])

    @TEST_TAG("ray")
    def test_ray_precise_dedup(self):
        tmp_yaml_file = osp.join(self.tmp_dir, 'config_1.yaml')
        tmp_out_path = osp.join(self.tmp_dir, 'output_dedup')
        text_keys = 'text'

        data_path = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__)))),
            'demos', 'data', 'demo-dataset-deduplication.jsonl')
        yaml_config = {
            'dataset_path': data_path,
            'executor_type': 'ray',
            'ray_address': 'auto',
            'text_keys': text_keys,
            'image_key': 'images',
            'export_path': tmp_out_path,
            'process': [
                {
                    'ray_document_deduplicator': {
                        'backend': 'ray_actor',
                    },
                }
            ]
        }

        with open(tmp_yaml_file, 'w') as file:
            yaml.dump(yaml_config, file)

        run_in_subprocess(f'python {self.script_path} --config {tmp_yaml_file}')

        self.assertTrue(osp.exists(tmp_out_path))

        jsonl_files = [os.path.join(tmp_out_path, f) \
                       for f in os.listdir(tmp_out_path) \
                        if f.endswith('.json')]
        data_cnt = 0
        for file in jsonl_files:
            with open(file, 'r') as f:
                for line in f.readlines():
                    if line.strip():
                        data_cnt += 1

        self.assertEqual(data_cnt, 13)

    @TEST_TAG("ray")
    def test_ray_minhash_dedup(self):
        tmp_yaml_file = osp.join(self.tmp_dir, 'config_2.yaml')
        tmp_out_path = osp.join(self.tmp_dir, 'output_minhash_dedup')
        text_keys = 'text'

        data_path = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__)))),
            'demos', 'data', 'demo-dataset-deduplication.jsonl')
        yaml_config = {
            'dataset_path': data_path,
            'executor_type': 'ray',
            'ray_address': 'auto',
            'text_keys': text_keys,
            'image_key': 'images',
            'export_path': tmp_out_path,
            'process': [
                {
                    'ray_bts_minhash_deduplicator': {
                        'lowercase': True,
                    },
                }
            ]
        }

        with open(tmp_yaml_file, 'w') as file:
            yaml.dump(yaml_config, file)

        run_in_subprocess(f'python {self.script_path} --config {tmp_yaml_file}')

        self.assertTrue(osp.exists(tmp_out_path))

        jsonl_files = [os.path.join(tmp_out_path, f) \
                       for f in os.listdir(tmp_out_path) \
                        if f.endswith('.json')]
        data_cnt = 0
        for file in jsonl_files:
            with open(file, 'r') as f:
                for line in f.readlines():
                    if line.strip():
                        data_cnt += 1

        self.assertEqual(data_cnt, 9)

    @TEST_TAG("ray")
    def test_ray_compute_stats_single_filter(self):
        tmp_yaml_file = osp.join(self.tmp_dir, 'config_3.yaml')
        tmp_out_path = osp.join(self.tmp_dir, 'output_3')
        text_keys = 'text'

        data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 '..', 'ops', 'data')
        ds_list = [{
            'audios': [os.path.join(data_path, 'audio1.wav')]  # about 6s
        }, {
            'audios': [os.path.join(data_path, 'audio2.wav')]  # about 14s
        }, {
            'audios': [os.path.join(data_path, 'audio3.ogg')]  # about 1min59s
        }]
        dataset_path = osp.join(self.tmp_dir, 'ds_list.jsonl')
        import json
        with open(dataset_path, 'w') as f:
            for data in ds_list:
                f.write(f'{json.dumps(data)}\n')
        yaml_config = {
            'dataset_path': dataset_path,
            'executor_type': 'ray',
            'ray_address': 'auto',
            'text_keys': text_keys,
            'image_key': 'images',
            'export_path': tmp_out_path,
            'process': [
                {
                    'audio_duration_filter': {
                        'max_duration': 10,
                    },
                }
            ]
        }

        with open(tmp_yaml_file, 'w') as file:
            yaml.dump(yaml_config, file)

        run_in_subprocess(f'python {self.script_path} --config {tmp_yaml_file}')

        self.assertTrue(osp.exists(tmp_out_path))

        jsonl_files = [os.path.join(tmp_out_path, f) \
                       for f in os.listdir(tmp_out_path) \
                        if f.endswith('.json')]
        data_cnt = 0
        for file in jsonl_files:
            with open(file, 'r') as f:
                for line in f.readlines():
                    if line.strip():
                        data_cnt += 1

        self.assertEqual(data_cnt, 1)

    @TEST_TAG("ray")
    def test_ray_compute_stats_batched_filter(self):
        tmp_yaml_file = osp.join(self.tmp_dir, 'config_4.yaml')
        tmp_out_path = osp.join(self.tmp_dir, 'output_4')
        text_key = 'text'

        ds_list = [{
            text_key: 'a=1\nb\nc=1+2+3+5\nd=6'
        }, {
            text_key:
            "Today is Sund Sund Sunda and it's a happy day!\nYou know"
        }, {
            text_key: 'a v s e e f g a qkc'
        }, {
            text_key: '，。、„" "«»１」「《》´∶：？！（）；–—．～…━〈〉【】％►'
        }, {
            text_key: 'Do you need a cup of coffee?'
        }, {
            text_key: 'emoji表情测试下😊，😸31231\n'
        }]
        dataset_path = osp.join(self.tmp_dir, 'ds_list.jsonl')
        import json
        with open(dataset_path, 'w') as f:
            for data in ds_list:
                f.write(f'{json.dumps(data)}\n')
        yaml_config = {
            'dataset_path': dataset_path,
            'executor_type': 'ray',
            'ray_address': 'auto',
            'text_keys': text_key,
            'image_key': 'images',
            'export_path': tmp_out_path,
            'process': [
                {
                    'average_line_length_filter': {
                        'min_len': 10,
                        'max_len': 20,
                        'batch_size': 3,
                    },
                }
            ]
        }

        with open(tmp_yaml_file, 'w') as file:
            yaml.dump(yaml_config, file)

        run_in_subprocess(f'python {self.script_path} --config {tmp_yaml_file}')

        self.assertTrue(osp.exists(tmp_out_path))

        jsonl_files = [os.path.join(tmp_out_path, f) \
                       for f in os.listdir(tmp_out_path) \
                        if f.endswith('.json')]
        data_cnt = 0
        for file in jsonl_files:
            with open(file, 'r') as f:
                for line in f.readlines():
                    if line.strip():
                        data_cnt += 1

        self.assertEqual(data_cnt, 2)


if __name__ == '__main__':
    unittest.main()
