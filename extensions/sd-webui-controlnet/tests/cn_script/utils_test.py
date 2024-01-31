import importlib
utils = importlib.import_module('extensions.sd-webui-controlnet.tests.utils', 'utils')


from scripts.utils import ndarray_lru_cache, get_unique_axis0

import unittest
import numpy as np

class TestNumpyLruCache(unittest.TestCase):

    def setUp(self):
        self.arr1 = np.array([1, 2, 3, 4, 5])
        self.arr2 = np.array([1, 2, 3, 4, 5])

    @ndarray_lru_cache(max_size=128)
    def add_one(self, arr):
        return arr + 1

    def test_same_array(self):
        # Test that the decorator works with numpy arrays.
        result1 = self.add_one(self.arr1)
        result2 = self.add_one(self.arr1)

        # If caching is working correctly, these should be the same object.
        self.assertIs(result1, result2)

    def test_different_array_same_data(self):
        # Test that the decorator works with different numpy arrays with the same data.
        result1 = self.add_one(self.arr1)
        result2 = self.add_one(self.arr2)

        # If caching is working correctly, these should be the same object.
        self.assertIs(result1, result2)

    def test_cache_size(self):
        # Test that the cache size limit is respected.
        arrs = [np.array([i]) for i in range(150)]

        # Add all arrays to the cache.
        
        result1 = self.add_one(arrs[0])
        for arr in arrs[1:]:
            self.add_one(arr)

        # Check that the first array is no longer in the cache.
        result2 = self.add_one(arrs[0])

        # If the cache size limit is working correctly, these should not be the same object.
        self.assertIsNot(result1, result2)

    def test_large_array(self):
        # Create two large arrays with the same elements in the beginning and end, but one different element in the middle.
        arr1 = np.ones(10000)
        arr2 = np.ones(10000)
        arr2[len(arr2)//2] = 0

        result1 = self.add_one(arr1)
        result2 = self.add_one(arr2)

        # If hashing is working correctly, these should not be the same object because the input arrays are not equal.
        self.assertIsNot(result1, result2)

class TestUniqueFunctions(unittest.TestCase):
    def test_get_unique_axis0(self):
        data = np.random.randint(0, 100, size=(100000, 3))
        data = np.concatenate((data, data))
        numpy_unique_res = np.unique(data, axis=0)
        get_unique_axis0_res = get_unique_axis0(data)
        self.assertEqual(np.array_equal(
            np.sort(numpy_unique_res, axis=0), np.sort(get_unique_axis0_res, axis=0),
        ), True)

if __name__ == '__main__':
    unittest.main()