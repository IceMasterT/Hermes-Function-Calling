import unittest
from unittest.mock import patch
import json
from typing import Dict, Any
from functions import get_current_crypto_price, get_crypto_market_data

class TestCryptoFunctions(unittest.TestCase):
    """Test cases for cryptocurrency-related functions"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are used for all test methods"""
        cls.valid_coins = ['bitcoin', 'ethereum', 'dogecoin']
        cls.mock_price_response = {
            'bitcoin': {'usd': 50000.00},
            'ethereum': {'usd': 3000.00},
            'dogecoin': {'usd': 0.25}
        }
        cls.mock_market_data = {
            'bitcoin': {
                'market_cap': 1000000000000,
                'total_volume': 50000000000,
                'price_change_24h': 1000.00,
                'market_cap_rank': 1,
                'high_24h': {'usd': 51000.00},
                'low_24h': {'usd': 49000.00}
            }
        }

    def setUp(self):
        """Set up test fixtures before each test method"""
        self.test_coin = 'bitcoin'
        self.test_currency = 'usd'

    def tearDown(self):
        """Clean up after each test method"""
        pass

    @patch('functions.requests.get')
    def test_get_current_crypto_price_success(self, mock_get):
        """Test successful price retrieval"""
        # Configure mock
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = self.mock_price_response

        # Test
        price = get_current_crypto_price(self.test_coin)
        
        # Assertions
        self.assertIsNotNone(price)
        self.assertIsInstance(price, float)
        self.assertEqual(price, 50000.00)
        mock_get.assert_called_once()

    @patch('functions.requests.get')
    def test_get_current_crypto_price_invalid_coin(self, mock_get):
        """Test price retrieval with invalid coin"""
        # Configure mock
        mock_get.return_value.status_code = 404
        
        # Test and assert
        with self.assertRaises(ValueError):
            get_current_crypto_price('invalid_coin')

    @patch('functions.requests.get')
    def test_get_current_crypto_price_api_error(self, mock_get):
        """Test handling of API errors"""
        # Configure mock
        mock_get.return_value.status_code = 500
        
        # Test and assert
        with self.assertRaises(ConnectionError):
            get_current_crypto_price(self.test_coin)

    @patch('functions.requests.get')
    def test_get_crypto_market_data_success(self, mock_get):
        """Test successful market data retrieval"""
        # Configure mock
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = self.mock_market_data

        # Test
        market_data = get_crypto_market_data(self.test_coin)
        
        # Assertions
        self.assertIsNotNone(market_data)
        self.assertIsInstance(market_data, dict)
        self.assertIn('market_cap', market_data)
        self.assertIn('total_volume', market_data)
        self.assertEqual(market_data['market_cap'], 1000000000000)
        self.assertEqual(market_data['total_volume'], 50000000000)
        mock_get.assert_called_once()

    @patch('functions.requests.get')
    def test_get_crypto_market_data_invalid_coin(self, mock_get):
        """Test market data retrieval with invalid coin"""
        # Configure mock
        mock_get.return_value.status_code = 404
        
        # Test and assert
        with self.assertRaises(ValueError):
            get_crypto_market_data('invalid_coin')

    @patch('functions.requests.get')
    def test_get_crypto_market_data_api_error(self, mock_get):
        """Test handling of API errors in market data retrieval"""
        # Configure mock
        mock_get.return_value.status_code = 500
        
        # Test and assert
        with self.assertRaises(ConnectionError):
            get_crypto_market_data(self.test_coin)

    def test_supported_coins(self):
        """Test that functions work with all supported coins"""
        for coin in self.valid_coins:
            with self.subTest(coin=coin):
                with patch('functions.requests.get') as mock_get:
                    # Configure mock
                    mock_get.return_value.status_code = 200
                    mock_get.return_value.json.return_value = {
                        coin: {'usd': 1000.00}
                    }
                    
                    # Test
                    price = get_current_crypto_price(coin)
                    
                    # Assertions
                    self.assertIsNotNone(price)
                    self.assertIsInstance(price, float)

    def test_input_validation(self):
        """Test input validation for both functions"""
        invalid_inputs = [None, "", " ", 123, 3.14, [], {}]
        
        for invalid_input in invalid_inputs:
            with self.subTest(invalid_input=invalid_input):
                with self.assertRaises(ValueError):
                    get_current_crypto_price(invalid_input)
                with self.assertRaises(ValueError):
                    get_crypto_market_data(invalid_input)

if __name__ == '__main__':
    unittest.main(verbosity=2)
