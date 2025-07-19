import unittest
from src.models import NLIModelTorch, SimilarityModel

class TestModels(unittest.TestCase):
    def test_load_nli_model(self):
        try:
            NLIModelTorch()
        except Exception as e:
            self.fail(f"NLIModelTorch failed to load with exception: {e}")

    def test_load_similarity_model(self):
        try:
            SimilarityModel()
        except Exception as e:
            self.fail(f"SimilarityModel failed to load with exception: {e}")

if __name__ == '__main__':
    unittest.main()
