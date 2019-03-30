import unittest
import pandas as pd
import load_data as loader

class TestLoadData(unittest.TestCase):

    def test_load_full_data(self):
        data = loader.load_full_data()
        train_data = pd.read_csv('./trainingset.csv')
        test_data = pd.read_csv('./testingset.csv')
        self.assertIsInstance(data, pd.core.frame.DataFrame)
        self.assertEqual(len(data.index), (len(train_data.index) + len(test_data.index)))
    
