import unittest
import pandas
import load_data as loader

class TestLoadData(unittest.TestCase):

    def test_load_full_data(self):
        data = loader.load_full_data()
        assertIsInstance(data, pandas.core.frame.DataFrame)
    
