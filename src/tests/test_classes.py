# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 10:04:45 2024

@author: huber
"""

import unittest
import numpy as np
from datetime import datetime, timedelta

from classes import Data, Metadata, Meas

# Zakładam, że klasy Data, Metadata i Meas są zaimportowane z Twojego projektu.
# from your_project_file import Data, Metadata, Meas


class TestData(unittest.TestCase):
    def setUp(self):
        self.x_data = np.array([1, 2, 3])
        self.y_data = np.array([4, 5, 6])
        self.data = Data(self.x_data, self.y_data)

    def test_initialization(self):
        self.assertTrue(np.array_equal(self.data.x_data, self.x_data))
        self.assertTrue(np.array_equal(self.data.y_data, self.y_data))

    def test_update(self):
        new_x_data = np.array([7, 8, 9])
        new_y_data = np.array([10, 11, 12])
        self.data.update(new_x_data, new_y_data)
        self.assertTrue(np.array_equal(self.data.x_data, new_x_data))
        self.assertTrue(np.array_equal(self.data.y_data, new_y_data))

    def test_copy(self):
        data_copy = self.data.copy()
        self.assertIsNot(self.data, data_copy)
        self.assertTrue(np.array_equal(self.data.x_data, data_copy.x_data))
        self.assertTrue(np.array_equal(self.data.y_data, data_copy.y_data))


class TestMetadata(unittest.TestCase):
    def setUp(self):
        self.starttime = datetime.now()
        self.endtime = self.starttime + timedelta(minutes=30)
        self.metadata = Metadata("test_name", "test_pair", 0.5, self.starttime, self.endtime)

    def test_initialization(self):
        self.assertEqual(self.metadata.name, "test_name")
        self.assertEqual(self.metadata.pair, "test_pair")
        self.assertEqual(self.metadata.shift, 0.5)
        self.assertEqual(self.metadata.starttime, self.starttime)
        self.assertEqual(self.metadata.endtime, self.endtime)
        self.assertAlmostEqual(self.metadata.duration_min, 30.0)

    def test_update(self):
        new_name = "updated_name"
        new_pair = "updated_pair"
        new_shift = 1.0
        new_starttime = self.starttime + timedelta(minutes=10)
        new_endtime = self.endtime + timedelta(minutes=10)
        self.metadata.update(name=new_name, pair=new_pair, shift=new_shift,
                             starttime=new_starttime, endtime=new_endtime)
        self.assertEqual(self.metadata.name, new_name)
        self.assertEqual(self.metadata.pair, new_pair)
        self.assertEqual(self.metadata.shift, new_shift)
        self.assertEqual(self.metadata.starttime, new_starttime)
        self.assertEqual(self.metadata.endtime, new_endtime)

    def test_copy(self):
        metadata_copy = self.metadata.copy()
        self.assertIsNot(self.metadata, metadata_copy)
        self.assertEqual(self.metadata.name, metadata_copy.name)
        self.assertEqual(self.metadata.pair, metadata_copy.pair)


class TestMeas(unittest.TestCase):
    def setUp(self):
        self.x_data = np.array([1, 2, 3])
        self.y_data = np.array([4, 5, 6])
        self.starttime = datetime.now()
        self.endtime = self.starttime + timedelta(minutes=30)
        self.meas = Meas(self.x_data, self.y_data, "test_name", "test_pair", 0.5,
                         self.starttime, self.endtime)

    def test_initialization(self):
        # Test Data initialization
        self.assertTrue(np.array_equal(self.meas.data.x_data, self.x_data))
        self.assertTrue(np.array_equal(self.meas.data.y_data, self.y_data))

        # Test Metadata initialization
        self.assertEqual(self.meas.metadata.name, "test_name")
        self.assertEqual(self.meas.metadata.pair, "test_pair")
        self.assertEqual(self.meas.metadata.shift, 0.5)
        self.assertEqual(self.meas.metadata.starttime, self.starttime)
        self.assertEqual(self.meas.metadata.endtime, self.endtime)

    def test_update_data(self):
        new_x_data = np.array([7, 8, 9])
        new_y_data = np.array([10, 11, 12])
        self.meas.update_data(new_x_data, new_y_data)
        self.assertTrue(np.array_equal(self.meas.data.x_data, new_x_data))
        self.assertTrue(np.array_equal(self.meas.data.y_data, new_y_data))

    def test_update_metadata(self):
        new_name = "updated_name"
        new_pair = "updated_pair"
        new_shift = 1.0
        new_starttime = self.starttime + timedelta(minutes=10)
        new_endtime = self.endtime + timedelta(minutes=10)
        self.meas.update_metadata(name=new_name, pair=new_pair, shift=new_shift,
                                  starttime=new_starttime, endtime=new_endtime)
        self.assertEqual(self.meas.metadata.name, new_name)
        self.assertEqual(self.meas.metadata.pair, new_pair)
        self.assertEqual(self.meas.metadata.shift, new_shift)
        self.assertEqual(self.meas.metadata.starttime, new_starttime)
        self.assertEqual(self.meas.metadata.endtime, new_endtime)

    def test_update(self):
        # Update both data and metadata
        new_x_data = np.array([7, 8, 9])
        new_y_data = np.array([10, 11, 12])
        new_name = "updated_name"
        new_pair = "updated_pair"
        new_shift = 1.0
        new_starttime = self.starttime + timedelta(minutes=10)
        new_endtime = self.endtime + timedelta(minutes=10)

        self.meas.update(new_x_data=new_x_data, new_y_data=new_y_data, name=new_name,
                         pair=new_pair, shift=new_shift, starttime=new_starttime, endtime=new_endtime)

        # Check data
        self.assertTrue(np.array_equal(self.meas.data.x_data, new_x_data))
        self.assertTrue(np.array_equal(self.meas.data.y_data, new_y_data))

        # Check metadata
        self.assertEqual(self.meas.metadata.name, new_name)
        self.assertEqual(self.meas.metadata.pair, new_pair)
        self.assertEqual(self.meas.metadata.shift, new_shift)
        self.assertEqual(self.meas.metadata.starttime, new_starttime)
        self.assertEqual(self.meas.metadata.endtime, new_endtime)

    def test_copy(self):
        meas_copy = self.meas.copy()
        self.assertIsNot(self.meas, meas_copy)
        self.assertTrue(np.array_equal(self.meas.data.x_data, meas_copy.data.x_data))
        self.assertTrue(np.array_equal(self.meas.data.y_data, meas_copy.data.y_data))
        self.assertEqual(self.meas.metadata.name, meas_copy.metadata.name)
        self.assertEqual(self.meas.metadata.pair, meas_copy.metadata.pair)


if __name__ == "__main__":
    unittest.main()
