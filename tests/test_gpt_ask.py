import unittest
from unittest.mock import patch
from src.app import create_app
from src.utils.err import NoPromptErr

class TestAskGPT(unittest.TestCase):
    """Test suite for GPT ask endpoint"""

    def setUp(self):
        """Set up test client and test environment"""
        self.app = create_app()
        self.client = self.app.test_client()
        self.client.testing = True

    @patch('src.routes.gpt_routes.query')
    def test_ask_endpoint_success(self, mock_openai):
        mock_openai.return_value ='Test response'
        
        response = self.client.post('/ask', json={'prompt': 'test'})
        self.assertEqual(response.status_code, 200)
        self.assertIn('Test response', response.get_json()['response'])

    def test_ask_endpoint_missing_prompt(self):
        """Test error handling when prompt is missing"""
        # Act
        response = self.client.post('/ask', json={})
        response_data = response.get_json()

        # Assert
        e = NoPromptErr()
        self.assertEqual(response.status_code, e.code)
        self.assertIn('error', response_data)

if __name__ == '__main__':
    unittest.main()