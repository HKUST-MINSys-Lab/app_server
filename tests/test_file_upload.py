import unittest
import os
import shutil
from unittest.mock import patch
from io import BytesIO
from src.app import create_app

from src.utils.err import (
    NoFilePartErr,
    NoPathSpecifiedErr,
    InvalidPathErr,
    PathAlreadyExistsErr
)

class TestUploadRoutes(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()
        self.test_dir = './uploads/test'
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_upload_success(self):
        data = {
            'file': (BytesIO(b'test content'), 'test.txt'),
            'path': 'test'
        }
        response = self.client.post(
            '/upload',
            data=data,
            content_type='multipart/form-data'
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['message'], 'File uploaded successfully')

    def test_upload_no_file(self):
        response = self.client.post(
            '/upload',
            data={'path': 'test'},
            content_type='multipart/form-data'
        )
        e = NoFilePartErr()
        self.assertEqual(response.status_code, e.code)

    def test_upload_no_path(self):
        data = {
            'file': (BytesIO(b'test content'), 'test.txt')
        }
        response = self.client.post(
            '/upload',
            data=data,
            content_type='multipart/form-data'
        )
        e = NoPathSpecifiedErr()
        self.assertEqual(response.status_code, e.code)

    def test_upload_invalid_path(self):
        data = {
            'file': (BytesIO(b'test content'), 'test.txt'),
            'path': '../test'
        }
        response = self.client.post(
            '/upload',
            data=data,
            content_type='multipart/form-data'
        )
        e = InvalidPathErr()
        self.assertEqual(response.status_code, e.code)

    def test_list_success(self):
        test_file = os.path.join(self.test_dir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write('test')
        
        response = self.client.get('/list?path=test')
        self.assertEqual(response.status_code, 200)
        self.assertIn('test.txt', response.json)

    def test_list_no_path(self):
        response = self.client.get('/list')
        e = NoPathSpecifiedErr()
        self.assertEqual(response.status_code, e.code)

    def test_list_invalid_path(self):
        response = self.client.get('/list?path=../test')
        e = InvalidPathErr()
        self.assertEqual(response.status_code, e.code)

    def test_mkdir_success(self):
        response = self.client.post(
            '/mkdir',
            json={'path': 'test/newdir'}
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'newdir')))

    def test_mkdir_no_path(self):
        response = self.client.post(
            '/mkdir',
            json={}
        )
        e = NoPathSpecifiedErr()
        self.assertEqual(response.status_code, e.code)

    def test_mkdir_invalid_path(self):
        response = self.client.post(
            '/mkdir',
            json={'path': '../test'}
        )
        e = InvalidPathErr()
        self.assertEqual(response.status_code, e.code)

    def test_mkdir_existing_path(self):
        os.makedirs(os.path.join(self.test_dir, 'existing'), exist_ok=True)
        response = self.client.post(
            '/mkdir',
            json={'path': 'test/existing'}
        )
        e = PathAlreadyExistsErr()
        self.assertEqual(response.status_code, e.code)

if __name__ == '__main__':
    unittest.main()