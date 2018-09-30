import unittest


class TestSample(unittest.TestCase):
    def setUp(self):
        pass

    def test_add(self):
        self.assertEqual((3 + 4), 7)
