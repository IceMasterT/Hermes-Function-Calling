import argparse
import torch
import json
import logging
from typing import Dict, List, Any, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    model_path: str
    chat_template: str
    load_in_4bit: bool
    max_new_tokens: int = 1500
    temperature: float = 0.8
    repetition_penalty: float = 1.1
    do_sample: bool = True

class PromptManager:
    def __init__(self, tools: List[Dict] = None):
        self.tools = tools or []
        self.few_shot_examples = self._load_few_shot_examples()

    def _load_few_shot_examples(self) -> List[Dict]:
        # Add your few-shot examples here
        return [
            {
                "role": "user",
                "content": "What's the current price of Bitcoin?",
                "expected_function": "get_crypto_price",
                "parameters": {"symbol": "BTC"}
            }
        ]

    def generate_prompt(self, chat_history: List[Dict], tools: List[Dict], num_fewshot: int = None) -> List[Dict]:
        prompt = []
        
        # Add few-shot examples if specified
        if num_fewshot:
            prompt.extend(self.few_shot_examples[:num_fewshot])
            
        # Add tools description
        tools_desc = "\n".join([f"{tool['name']}: {tool['description']}" for tool in tools])
        prompt.append({"role": "system", "content": f"Available tools:\n{tools_desc}"})
        
        # Add chat history
        prompt.extend(chat_history)
        
        return prompt

class ModelInference:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.prompter = PromptManager()
        self.model, self.tokenizer = self._initialize_model()
        logger.info(f"Model initialized with config: {self.model.config}")

    def _initialize_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        # Initialize BitsAndBytes config if needed
        bnb_config = None
        if self.config.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
            return_dict=True,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        # Set chat template if not defined
        if tokenizer.chat_template is None:
            tokenizer.chat_template = self.config.chat_template

        return model, tokenizer

    def run_inference(self, prompt: List[Dict]) -> str:
        # Tokenize input
        inputs = self.tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            return_tensors='pt'
        )

        # Generate response
        tokens = self.model.generate(
            inputs.to(self.model.device),
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            repetition_penalty=self.config.repetition_penalty,
            do_sample=self.config.do_sample,
            eos_token_id=self.tokenizer.eos_token_id
        )

        # Decode response
        completion = self.tokenizer.decode(
            tokens[0], 
            skip_special_tokens=False,
            clean_up_tokenization_space=True
        )

        return completion

    def process_completion(self, completion: str) -> Tuple[Optional[List[Dict]], str, Optional[str]]:
        """Process the model completion and extract function calls."""
        try:
            # Extract assistant message
            assistant_message = self._extract_assistant_message(completion)
            if not assistant_message:
                return None, "", "Failed to extract assistant message"

            # Extract and validate function calls
            tool_calls = self._extract_tool_calls(assistant_message)
            if not tool_calls:
                return None, assistant_message, None

            return tool_calls, assistant_message, None

        except Exception as e:
            logger.error(f"Error processing completion: {e}")
            return None, "", str(e)

    def execute_function_call(self, tool_call: Dict) -> str:
        """Execute a function call and return the result."""
        try:
            function_name = tool_call.get("name")
            function_args = tool_call.get("arguments", {})
            
            # Import and execute function dynamically
            module = __import__('functions')
            function = getattr(module, function_name)
            
            result = function(**function_args)
            return json.dumps({"name": function_name, "content": result})

        except Exception as e:
            logger.error(f"Error executing function {tool_call.get('name')}: {e}")
            return json.dumps({"error": str(e)})

    def generate_function_call(self, query: str, max_depth: int = 5) -> str:
        """Generate and execute function calls recursively."""
        try:
            depth = 0
            chat_history = [{"role": "user", "content": query}]
            tools = self.prompter.tools

            while depth < max_depth:
                # Generate prompt and get completion
                prompt = self.prompter.generate_prompt(chat_history, tools)
                completion = self.run_inference(prompt)

                # Process completion
                tool_calls, assistant_message, error = self.process_completion(completion)
                chat_history.append({"role": "assistant", "content": assistant_message})

                if error:
                    logger.error(f"Error at depth {depth}: {error}")
                    break

                if not tool_calls:
                    break

                # Execute function calls and add results to chat history
                for tool_call in tool_calls:
                    result = self.execute_function_call(tool_call)
                    chat_history.append({"role": "function", "name": tool_call["name"], "content": result})

                depth += 1

            return chat_history[-1]["content"]

        except Exception as e:
            logger.error(f"Error in generate_function_call: {e}")
            return str(e)

def main():
    parser = argparse.ArgumentParser(description="Run function call inference")
    parser.add_argument("--model_path", type=str, default="NousResearch/Hermes-2-Pro-Llama-3-8B")
    parser.add_argument("--chat_template", type=str, default="chatml")
    parser.add_argument("--load_in_4bit", type=str, default="False")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--max_depth", type=int, default=5)
    args = parser.parse_args()

    # Create model config
    config = ModelConfig(
        model_path=args.model_path,
        chat_template=args.chat_template,
        load_in_4bit=args.load_in_4bit.lower() == "true"
    )

    # Initialize and run inference
    inference = ModelInference(config)
    result = inference.generate_function_call(args.query, args.max_depth)
    print(f"Final result: {result}")

if __name__ == "__main__":
    main()
