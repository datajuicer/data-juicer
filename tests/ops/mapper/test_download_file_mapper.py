import unittest
import os
import os.path as osp
import shutil
import tempfile
import threading
import numpy as np
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.utils.mm_utils import load_image, load_image_byte
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase
from data_juicer.ops.mapper.download_file_mapper import DownloadFileMapper


class DownloadFileMapperTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()

        self.temp_dir = tempfile.mkdtemp()
        self.data_path = osp.abspath(osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data'))
        self.img1_path = osp.join(self.data_path, 'img1.png')
        self.img2_path = osp.join(self.data_path, 'img2.jpg')
        self.img3_path = osp.join(self.data_path, 'img3.jpg')

        # start HTTP server
        self.server_address = ('localhost', 0)  # 0 means random port
        self.httpd = HTTPServer(
            self.server_address,
            partial(SimpleHTTPRequestHandler, directory=self.data_path)
        )
        self.port = self.httpd.server_address[1]
        self.img1_url = f'http://localhost:{self.port}/{os.path.basename(self.img1_path)}'
        self.img2_url = f'http://localhost:{self.port}/{os.path.basename(self.img2_path)}'
        self.img3_url = f'http://localhost:{self.port}/{os.path.basename(self.img3_path)}'
        
        # start the server in a thread
        self.server_thread = threading.Thread(target=self.httpd.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        shutil.rmtree(self.temp_dir)

        super().tearDown()

    def _test_image_download(self, ds_list, save_field=None):
        op = DownloadFileMapper(
                save_dir=self.temp_dir,
                download_field='images',
                save_field=save_field)

        dataset = Dataset.from_list(ds_list)
        dataset = dataset.map(op.process, batch_size=2)
        
        res_list = dataset.to_list()
        res_list = sorted(res_list, key=lambda x: x['id'])

        self.assertEqual(len(ds_list), len(res_list))

        for i in range(len(ds_list)):
            source, res = ds_list[i], res_list[i]
            for j in range(len(source[op.image_key])):
                s_path, r_path = source[op.image_key][j], res[op.image_key][j]
                fname = os.path.basename(s_path)
                self.assertEqual(fname, os.path.basename(r_path))
                if s_path.startswith('http'):
                    self.assertEqual(os.path.dirname(r_path), self.temp_dir)
                else:
                    self.assertEqual(s_path, r_path)

                t_img = np.array(load_image(os.path.join(self.data_path, fname)))
                r_img = np.array(load_image(r_path))

                np.testing.assert_array_equal(t_img, r_img)

                if save_field:
                    self.assertEqual(
                        res[save_field][j],
                        load_image_byte(os.path.join(self.data_path, fname))
                    )

    def test_image_download(self):
        ds_list = [{
            'images': [self.img1_url],
            'id': 1
        }, {
            'images': [self.img2_url, self.img3_url],
            'id': 2
        }, {
            'images': [self.img1_url, self.img2_url, self.img3_url],
            'id': 3
        }]

        self._test_image_download(ds_list)
        
    def test_image_url_and_local_path(self):
        ds_list = [{
            'images': [self.img1_path],
            'id': 1
        }, {
            'images': [self.img2_path, self.img3_url],
            'id': 2
        }, {
            'images': [self.img1_path, self.img2_url, self.img3_path],
            'id': 3
        }]

        self._test_image_download(ds_list)
        
    def test_download_image_failed(self):
        ds_list = [{
            'images': self.img2_url + '_failed_test',
            'id': 1
        }, {
            'images': self.img3_url + '_failed_test',
            'id': 2
        }, {
            'images': self.img1_url,
            'id': 3
        }]

        op = DownloadFileMapper(
                save_dir=self.temp_dir,
                download_field='images')

        dataset = Dataset.from_list(ds_list)
        dataset = dataset.map(op.process, batch_size=2)
        
        res_list = dataset.to_list()
        res_list = sorted(res_list, key=lambda x: x['id'])

        self.assertEqual(len(ds_list), len(res_list))

        for i in range(len(ds_list)):
            source, res = ds_list[i], res_list[i]
            s_path, r_path = source[op.image_key], res[op.image_key]
            fname = os.path.basename(s_path)
            self.assertEqual(fname, os.path.basename(r_path))
            if s_path.startswith('http') and 'failed_test' not in s_path:
                self.assertEqual(os.path.dirname(r_path), self.temp_dir)
            else:
                self.assertEqual(s_path, r_path)

    def test_image_str_type(self):
        ds_list = [{
            'images': self.img2_path,
            'id': 1
        }, {
            'images': self.img3_path,
            'id': 2
        }, {
            'images': self.img1_url,
            'id': 3
        }]

        op = DownloadFileMapper(
                save_dir=self.temp_dir,
                download_field='images')

        dataset = Dataset.from_list(ds_list)
        dataset = dataset.map(op.process, batch_size=2)
        
        res_list = dataset.to_list()
        res_list = sorted(res_list, key=lambda x: x['id'])

        self.assertEqual(len(ds_list), len(res_list))

        for i in range(len(ds_list)):
            source, res = ds_list[i], res_list[i]
            s_path, r_path = source[op.image_key], res[op.image_key]
            fname = os.path.basename(s_path)
            self.assertEqual(fname, os.path.basename(r_path))
            if s_path.startswith('http'):
                self.assertEqual(os.path.dirname(r_path), self.temp_dir)
            else:
                self.assertEqual(s_path, r_path)

            t_img = np.array(load_image(os.path.join(self.data_path, fname)))
            r_img = np.array(load_image(r_path))

            np.testing.assert_array_equal(t_img, r_img)

    def test_image_with_only_save_field(self):
        ds_list = [{
            'images': [self.img1_url],
            'id': 1
        }, {
            'images': [self.img2_url, self.img3_url],
            'id': 2
        }, {
            'images': [self.img1_url, self.img2_url, self.img3_url],
            'id': 3
        }, {
            'images': [self.img2_url],
            'id': 4
        },
        ]

        save_field='image_bytes'

        op = DownloadFileMapper(
                save_dir=None,
                download_field='images',
                save_field=save_field)

        dataset = Dataset.from_list(ds_list)
        dataset = dataset.map(op.process, batch_size=2)
        
        res_list = dataset.to_list()
        res_list = sorted(res_list, key=lambda x: x['id'])

        self.assertEqual(len(ds_list), len(res_list))

        for i in range(len(ds_list)):
            source, res = ds_list[i], res_list[i]
            for j in range(len(source[op.image_key])):
                s_path, r_path = source[op.image_key][j], res[op.image_key][j]
                self.assertEqual(s_path, r_path)
                fname = os.path.basename(s_path)
                self.assertEqual(
                    res[save_field][j],
                    load_image_byte(os.path.join(self.data_path, fname))
                )

    def test_image_with_savefield_and_savedir(self):
        ds_list = [{
            'images': [self.img1_url],
            'id': 1
        }, {
            'images': [self.img2_path, self.img3_url],
            'id': 2
        }, {
            'images': [self.img1_url, self.img2_path, self.img3_url],
            'id': 3
        }
        ]
        
        self._test_image_download(ds_list, save_field='image_bytes')

    def test_image_with_savefield_and_resume(self):
        save_field='image_bytes'

        ds_list = [{
            'images': [self.img1_url],
            'id': 1,
            save_field: []
        }, {
            'images': [self.img2_url, self.img3_url],
            'id': 2,
            save_field: ['loaded', None]
        }, {
            'images': [self.img1_url, self.img2_path, self.img3_url],
            'id': 3
        }, {
            'images': [self.img2_url],
            'id': 4,
            save_field: ['loaded', None]  # will be fixed auto
        }]


        tgt_list = [{
            'images': [self.img1_url],
            'id': 1,
            save_field: [load_image_byte(self.img1_path)]
        }, {
            'images': [self.img2_url, self.img3_url],
            'id': 2,
            save_field: [b'loaded', load_image_byte(self.img3_path)],
        }, {
            'images': [self.img1_url, self.img2_path, self.img3_url],
            'id': 3,
            save_field: [
                load_image_byte(self.img1_path),
                load_image_byte(self.img2_path),
                load_image_byte(self.img3_path)]
        }, {
            'images': [self.img2_url],
            'id': 4,
            save_field: [load_image_byte(self.img2_path)]
        }]

        op = DownloadFileMapper(
                save_dir=None,
                download_field='images',
                save_field=save_field,
                resume_download=True)

        dataset = Dataset.from_list(ds_list)
        dataset = dataset.map(op.process, batch_size=2)
        
        res_list = dataset.to_list()
        res_list = sorted(res_list, key=lambda x: x['id'])

        self.assertEqual(len(ds_list), len(res_list))

        for i in range(len(ds_list)):
            self.assertListEqual(res_list[i][save_field], tgt_list[i][save_field])
            self.assertEqual(res_list[i]['id'], tgt_list[i]['id'])
            self.assertListEqual(res_list[i]['images'], tgt_list[i]['images'])

    def test_image_with_savefield_and_resume_and_savedir(self):

        def _to_tmp_path(img_path):
            return osp.join(self.temp_dir, osp.basename(img_path))

        ds_list = [{
            'images': [self.img1_url],
            'id': 1,
            'image_bytes': []
        }, {
            'images': [self.img2_url, self.img3_url],
            'id': 2,
            'image_bytes': ['loaded', None]
        }, {
            'images': [self.img1_url, self.img2_path, self.img3_url],
            'id': 3
        }, {
            'images': [self.img2_url],
            'id': 4,
            'image_bytes': ['loaded', None]  # will be fixed auto
        }]


        tgt_list = [{
            'images': [_to_tmp_path(self.img1_url)],
            'id': 1,
            'image_bytes': [load_image_byte(self.img1_path)]
        }, {
            'images': [_to_tmp_path(self.img2_url), _to_tmp_path(self.img3_url)],
            'id': 2,
            'image_bytes': [b'loaded', load_image_byte(self.img3_path)],
        }, {
            'images': [
                _to_tmp_path(self.img1_url),
                self.img2_path,
                _to_tmp_path(self.img3_url)],
            'id': 3,
            'image_bytes': [
                load_image_byte(self.img1_path),
                load_image_byte(self.img2_path),
                load_image_byte(self.img3_path)]
        }, {
            'images': [_to_tmp_path(self.img2_url)],
            'id': 4,
            'image_bytes': [load_image_byte(self.img2_path)]
        }]

        op = DownloadFileMapper(
                save_dir=self.temp_dir,
                download_field='images',
                save_field='image_bytes',
                resume_download=True)

        dataset = Dataset.from_list(ds_list)
        dataset = dataset.map(op.process, batch_size=2)
        
        res_list = dataset.to_list()
        res_list = sorted(res_list, key=lambda x: x['id'])

        self.assertEqual(len(ds_list), len(res_list))

        for i in range(len(ds_list)):
            self.assertListEqual(res_list[i]['image_bytes'], tgt_list[i]['image_bytes'])
            self.assertEqual(res_list[i]['id'], tgt_list[i]['id'])
            self.assertListEqual(res_list[i]['images'], tgt_list[i]['images'])


if __name__ == '__main__':
    unittest.main()
