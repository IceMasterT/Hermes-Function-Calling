import unittest
from functions import get_current_crypto_price, get_crypto_market_data

class TestCryptoFunctions(unittest.TestCase):
    def test_get_current_crypto_price(self):
        price = get_current_crypto_price('bitcoin')
        self.assertIsNotNone(price)
        self.assertGreater(price, 0)

    def test_get_crypto_market_data(self):
        market_data = get_crypto_market_data('bitcoin')
        self.assertIsNotNone(market_data)
        self.assertIn('market_cap', market_data)
        self.assertIn('total_volume', market_data)

if __name__ == "__main__":
    unittest.main()
