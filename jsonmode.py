import argparse
import torch
import json
import logging
from typing import List, Optional, Dict, Tuple, Any
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic Model Definition
class Character(BaseModel):
    name: str
    species: str
    role: str
    personality_traits: Optional[List[str]]
    special_attacks: Optional[List[str]]

    class Config:
        schema_extra = {
            "additionalProperties": False
        }

@dataclass
class ModelConfig:
    model_path: str
    chat_template: str
    load_in_4bit: bool
    max_new_tokens: int = 1500
    temperature: float = 0.8
    repetition_penalty: float = 1.1
    do_sample: bool = True

class JsonModeInference:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model, self.tokenizer = self._initialize_model()
        self.pydantic_schema = Character.schema_json()
        logger.info(f"Model initialized with config: {self.model.config}")
        logger.info(f"Using JSON schema:\n{json.dumps(json.loads(self.pydantic_schema), indent=2)}")

    def _initialize_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Initialize the model and tokenizer."""
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

    def run_inference(self, prompt: List[Dict[str, str]]) -> str:
        """Run model inference with the given prompt."""
        try:
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

        except Exception as e:
            logger.error(f"Error in run_inference: {e}")
            raise

    def extract_json_from_completion(self, completion: str) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """Extract and validate JSON from the model completion."""
        try:
            # Try to parse the completion as JSON
            try:
                json_str = completion.strip()
                # Find the first { and last }
                start = json_str.find('{')
                end = json_str.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = json_str[start:end]
                json_data = json.loads(json_str)
            except json.JSONDecodeError as e:
                return False, None, f"JSON parsing error: {str(e)}"

            # Validate against schema
            try:
                validated_data = Character(**json_data).dict()
                return True, validated_data, None
            except Exception as e:
                return False, None, f"Schema validation error: {str(e)}"

        except Exception as e:
            logger.error(f"Error in extract_json_from_completion: {e}")
            return False, None, str(e)

    def generate_json_completion(self, query: str, max_depth: int = 5) -> Optional[Dict]:
        """Generate JSON completion with recursive refinement."""
        try:
            depth = 0
            # Create system prompt with schema
            sys_prompt = (
                "You are a helpful assistant that answers in JSON. "
                f"Here's the json schema you must adhere to:\n<schema>\n{self.pydantic_schema}\n</schema>"
            )
            
            # Initialize conversation
            conversation = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": query}
            ]

            while depth < max_depth:
                # Get model completion
                completion = self.run_inference(conversation)
                
                # Extract and validate JSON
                is_valid, json_data, error = self.extract_json_from_completion(completion)
                
                if is_valid:
                    logger.info(f"Valid JSON generated:\n{json.dumps(json_data, indent=2)}")
                    return json_data
                
                # If invalid, add error message and try again
                error_message = (
                    f"Previous attempt failed. Error: {error}\n"
                    "Please try again and ensure the response is valid JSON matching the schema."
                )
                conversation.append({"role": "user", "content": error_message})
                
                depth += 1
                logger.info(f"Attempt {depth}/{max_depth} failed, trying again...")

            logger.warning(f"Failed to generate valid JSON after {max_depth} attempts")
            return None

        except Exception as e:
            logger.error(f"Error in generate_json_completion: {e}")
            return None

def main():
    parser = argparse.ArgumentParser(description="Run JSON mode inference")
    parser.add_argument("--model_path", type=str, default="NousResearch/Hermes-2-Pro-Llama-3-8B")
    parser.add_argument("--chat_template", type=str, default="chatml")
    parser.add_argument("--load_in_4bit", type=str, default="False")
    parser.add_argument("--query", type=str, default="Please return a json object to represent Goku from the anime Dragon Ball Z?")
    parser.add_argument("--max_depth", type=int, default=5)
    args = parser.parse_args()

    # Create model config
    config = ModelConfig(
        model_path=args.model_path,
        chat_template=args.chat_template,
        load_in_4bit=args.load_in_4bit.lower() == "true"
    )

    # Initialize and run inference
    inference = JsonModeInference(config)
    result = inference.generate_json_completion(args.query, args.max_depth)
    
    if result:
        print("\nGenerated JSON:")
        print(json.dumps(result, indent=2))
    else:
        print("\nFailed to generate valid JSON")

if __name__ == "__main__":
    main()
