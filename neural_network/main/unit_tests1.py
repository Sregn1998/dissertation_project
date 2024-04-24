import unittest
import sys
sys.path.append("..") 
from predict import DepressionDetector

class TestDepressionDetector(unittest.TestCase):
    
    def setUp(self):
        self.detector = DepressionDetector(model_path="neural_network/depression_model.h5")
    
    def test_preprocess_text(self):
        text = "Hello, World!"
        preprocessed_text = self.detector.preprocess_text(text)
        self.assertEqual(preprocessed_text, "hello world")
    
    def test_read_text_from_file(self):
        file_path = "test_file.txt"
        with open(file_path, "w", encoding="utf-8") as file:
            file.write("Test text")
        text = self.detector.read_text_from_file(file_path)
        self.assertEqual(text, "Test text")
    
    def test_predict_depression(self):
        text = "I feel sad and hopeless."
        average_prediction = self.detector.predict_depression(text)
        self.assertTrue(0 <= average_prediction <= 1)
    
    def test_assess_mental_health(self):
        high_prediction = 0.8
        low_prediction = 0.4
        self.assertEqual(self.detector.assess_mental_health(high_prediction), "Всё ок.")
        self.assertEqual(self.detector.assess_mental_health(low_prediction), "У вас наблюдаются расстройства психического характера. Рекомендую вам обратиться к специалисту за помощью.")

if __name__ == "__main__":
    unittest.main()
