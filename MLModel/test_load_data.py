import unittest
import pandas as pd
import data_handler as handler

class TestLoadData(unittest.TestCase):

    def test_load_full_data(self):
        data = handler.load_full_data()
        train_data = pd.read_csv('./trainingset.csv')
        test_data = pd.read_csv('./testingset.csv')
        self.assertIsInstance(data, pd.core.frame.DataFrame)
        self.assertEqual(len(data.index), (len(train_data.index) + len(test_data.index)))
