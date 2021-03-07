"""
test.py

This file tests the various necessary util functions.
"""
from .io import *
import contextlib
import pandas as pd
import unittest


@contextlib.contextmanager
def set_env(environ):
    """
    Temporarily set the process environment variables. Taken from https://stackoverflow.com/questions/2059482/python-temporarily-modify-the-current-processs-environment/34333710

    >>> with set_env(PLUGINS_DIR=u'test/plugins'):
    ...   "PLUGINS_DIR" in os.environ
    True

    >>> "PLUGINS_DIR" in os.environ
    False

    :type environ: dict[str, unicode]
    :param environ: Environment variables to set
    """
    old_environ = dict(os.environ)
    os.environ.update(environ)
    print(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_environ)


class IOTests(unittest.TestCase):

    def setUp(self):
        self.toy_df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [2, 4, 6]})
        self.fake_creds = {
            'AWS_ACCESS_KEY_ID': 'nothing',
            'AWS_SECRET_ACCESS_KEY': 'nothing',
            'AWS_DEFAULT_REGION': 'nothing',
        }

    def test_write_file_without_creds(self):
        pass

    def test_write_file_without_name(self):
        with self.assertRaises(AssertionError):
            write_file(self.toy_df, '')

    def test_write_file_scratch(self):
        filename = write_file(self.toy_df, 'test.pq')
        self.assertEqual(
            filename, 's3://toy-applied-ml-pipeline/scratch/test.pq')

    def test_write_file_no_scratch(self):
        filename = write_file(self.toy_df, 'test/test.pq', scratch=False)
        self.assertEqual(
            filename, 's3://toy-applied-ml-pipeline/test/test.pq')

    def test_save_output_without_creds(self):
        pass

    def test_save_output_no_version(self):
        filename = save_output_df(self.toy_df, 'test')
        self.assertIn('s3://toy-applied-ml-pipeline/dev/test/', filename)

    def test_save_output_no_component(self):
        with self.assertRaises(AssertionError):
            save_output_df(self.toy_df, '')

    def test_save_output_with_version(self):
        filename = save_output_df(self.toy_df, 'test',
                                  version='test', overwrite=True)
        self.assertEqual(
            filename, 's3://toy-applied-ml-pipeline/dev/test/test.pq')

    def test_save_output_no_overwrite(self):
        save_output_df(self.toy_df, 'test',
                       version='test_no_overwrite', overwrite=True)
        with self.assertRaises(OSError):
            save_output_df(self.toy_df, 'test', version='test_no_overwrite')

    def test_save_output_overwrite(self):
        filename_1 = save_output_df(self.toy_df, 'test',
                                    version='test_overwrite', overwrite=True)
        filename_2 = save_output_df(self.toy_df, 'test',
                                    version='test_overwrite', overwrite=True)
        self.assertEqual(filename_1, filename_2)


if __name__ == '__main__':
    unittest.main()
