import os
import yaml
import json
import datetime
import requests
from typing import List, Dict, Optional
from pydantic import BaseModel
from langchain.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PromptSchema(BaseModel):
    """Schema for prompt structure"""
    role: str
    objective: str
    tools: str
    examples: Optional[str]
    schema: str
    instructions: str

class CryptoFunction:
    """Cryptocurrency-related functions"""
    
    @staticmethod
    @tool
    def get_current_crypto_price(symbol: str) -> float:
        """
        Get the current cryptocurrency price for a given symbol.
        
        Args:
            symbol (str): The cryptocurrency symbol (e.g., 'bitcoin' for Bitcoin)
            
        Returns:
            float: Current price in USD
        """
        try:
            url = f'https://api.coingecko.com/api/v3/simple/price?ids={symbol}&vs_currencies=usd'
            response = requests.get(url)
            data = response.json()
            return data[symbol]['usd']
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            return None

    @staticmethod
    @tool
    def get_crypto_market_data(symbol: str) -> dict:
        """
        Get market data for a given cryptocurrency.
        
        Args:
            symbol (str): The cryptocurrency symbol (e.g., 'bitcoin' for Bitcoin)
            
        Returns:
            dict: Market data including market cap, volume, and price changes
        """
        try:
            url = f'https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&ids={symbol}'
            response = requests.get(url)
            data = response.json()[0]
            return {
                'symbol': symbol,
                'market_cap': data.get('market_cap'),
                'total_volume': data.get('total_volume'),
                'price_change_24h': data.get('price_change_percentage_24h'),
                'price_change_7d': data.get('price_change_percentage_7d')
            }
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return {}

class PromptManager:
    """Manages prompt generation and tool management"""
    
    def __init__(self, custom_prompt_path: Optional[str] = None):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.prompt_path = custom_prompt_path or os.path.join(self.script_dir, 'prompt_assets', 'sys_prompt.yml')
        self.tools = self._initialize_tools()
        self.few_shot_examples = self._load_few_shot_examples()

    def _initialize_tools(self) -> List[dict]:
        """Initialize cryptocurrency tools"""
        functions = [
            CryptoFunction.get_current_crypto_price,
            CryptoFunction.get_crypto_market_data
        ]
        return [convert_to_openai_tool(f) for f in functions]

    def _load_few_shot_examples(self) -> List[Dict]:
        """Load few-shot examples for prompt generation"""
        return [
            {
                "user": "What's the current price of Bitcoin?",
                "assistant": "Let me check the current Bitcoin price for you.",
                "function": "get_current_crypto_price",
                "parameters": {"symbol": "bitcoin"}
            },
            {
                "user": "Show me market data for Ethereum",
                "assistant": "I'll fetch the market data for Ethereum.",
                "function": "get_crypto_market_data",
                "parameters": {"symbol": "ethereum"}
            }
        ]

    def _load_prompt_schema(self) -> PromptSchema:
        """Load prompt schema from YAML file"""
        try:
            with open(self.prompt_path, 'r') as file:
                yaml_content = yaml.safe_load(file)
            return PromptSchema(**yaml_content)
        except Exception as e:
            logger.error(f"Error loading prompt schema: {e}")
            raise

    def _format_prompt(self, schema: PromptSchema, variables: Dict) -> str:
        """Format prompt according to schema and variables"""
        try:
            formatted_prompt = []
            for field, value in schema.dict().items():
                if field == "examples" and not variables.get("examples"):
                    continue
                formatted_value = value.format(**variables)
                if field == "instructions":
                    formatted_prompt.append(formatted_value)
                else:
                    formatted_value = formatted_value.replace("\n", " ")
                    formatted_prompt.append(f"{field}: {formatted_value}")
            return "\n".join(formatted_prompt)
        except Exception as e:
            logger.error(f"Error formatting prompt: {e}")
            raise

    def generate_prompt(self, 
                       query: str, 
                       num_examples: int = 2, 
                       include_schema: bool = True) -> List[Dict[str, str]]:
        """
        Generate a complete prompt with system context and user query
        
        Args:
            query (str): User's query
            num_examples (int): Number of few-shot examples to include
            include_schema (bool): Whether to include function schema
            
        Returns:
            List[Dict[str, str]]: Formatted prompt messages
        """
        try:
            # Load prompt schema
            schema = self._load_prompt_schema()
            
            # Prepare variables for prompt formatting
            variables = {
                "date": datetime.datetime.now().strftime("%Y-%m-%d"),
                "tools": json.dumps(self.tools, indent=2) if include_schema else "",
                "examples": self.few_shot_examples[:num_examples] if num_examples > 0 else None,
                "query": query
            }
            
            # Format system prompt
            system_prompt = self._format_prompt(schema, variables)
            
            # Construct final prompt
            prompt = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
            
            logger.info(f"Generated prompt with {num_examples} examples")
            return prompt

        except Exception as e:
            logger.error(f"Error generating prompt: {e}")
            raise

    def get_tool_schema(self) -> Dict:
        """Get the schema for available tools"""
        return {tool["name"]: tool["parameters"] for tool in self.tools}

    def validate_tool_call(self, tool_name: str, parameters: Dict) -> bool:
        """Validate a tool call against the schema"""
        tool_schema = self.get_tool_schema().get(tool_name)
        if not tool_schema:
            return False
        try:
            # Basic parameter validation
            required_params = {
                p["name"] for p in tool_schema.get("properties", {}).values()
                if p.get("required", False)
            }
            return all(param in parameters for param in required_params)
        except Exception as e:
            logger.error(f"Error validating tool call: {e}")
            return False

# Example usage
if __name__ == "__main__":
    # Initialize prompt manager
    prompt_manager = PromptManager()
    
    # Generate a prompt
    query = "What's the current price of Bitcoin and its 24-hour change?"
    prompt = prompt_manager.generate_prompt(query, num_examples=1)
    
    # Print generated prompt
    print("\nGenerated Prompt:")
    for message in prompt:
        print(f"\n{message['role'].upper()}:")
        print(message['content'])
    
    # Example tool call validation
    tool_call = {
        "name": "get_crypto_market_data",
        "parameters": {"symbol": "bitcoin"}
    }
    is_valid = prompt_manager.validate_tool_call(
        tool_call["name"], 
        tool_call["parameters"]
    )
    print(f"\nTool call validation: {'Valid' if is_valid else 'Invalid'}")
