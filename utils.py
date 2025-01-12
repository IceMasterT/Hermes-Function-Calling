import os
import re
import ast
import json
import yaml
import logging
import datetime
from typing import Dict, Any, Tuple, Optional, List
from logging.handlers import RotatingFileHandler
import xml.etree.ElementTree as ET
from art import text2art

class LoggerSetup:
    """Handles logging configuration and management"""
    
    @staticmethod
    def setup_logger(name: str, log_dir: str = "logs") -> logging.Logger:
        """
        Set up a logger with both file and console handlers.
        
        Args:
            name: Name of the logger
            log_dir: Directory for log files
            
        Returns:
            logging.Logger: Configured logger instance
        """
        # Create logs directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S'
        )
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        file_path = os.path.join(
            log_dir, 
            f"{name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
        )
        file_handler = RotatingFileHandler(
            file_path,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger

class TextProcessor:
    """Handles text processing and extraction tasks"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.chat_templates = {
            "zephyr": r'<\|assistant\|>((?:(?!<\|assistant\|>).)*)$',
            "chatml": r'<\|im_start\|>\s*assistant((?:(?!<\|im_start\|>\s*assistant).)*)$',
            "vicuna": r'ASSISTANT:\s*((?:(?!ASSISTANT:).)*)$'
        }

    def get_assistant_message(self, 
                            completion: str, 
                            chat_template: str, 
                            eos_token: str) -> Optional[str]:
        """
        Extract assistant's message from completion text.
        
        Args:
            completion: Raw completion text
            chat_template: Template type to use
            eos_token: End of sequence token
            
        Returns:
            Optional[str]: Extracted assistant message or None if extraction fails
        """
        try:
            completion = completion.strip()
            
            if chat_template not in self.chat_templates:
                raise ValueError(f"Unsupported chat template: {chat_template}")
                
            pattern = self.chat_templates[chat_template]
            assistant_match = re.search(pattern, completion, re.DOTALL)
            
            if assistant_match:
                content = assistant_match.group(1).strip()
                if chat_template == "vicuna":
                    eos_token = f"</s>{eos_token}"
                return content.replace(eos_token, "")
            
            self.logger.warning("No match found for assistant pattern")
            return None
            
        except Exception as e:
            self.logger.error(f"Error extracting assistant message: {e}")
            return None

    def extract_tool_calls(self, content: str) -> Tuple[bool, List[Dict], Optional[str]]:
        """
        Extract and validate tool calls from content.
        
        Args:
            content: Content containing tool calls
            
        Returns:
            Tuple containing validation status, tool calls list, and error message
        """
        try:
            # Wrap content in root element
            xml_content = f"<root>{content}</root>"
            root = ET.fromstring(xml_content)
            
            tool_calls = []
            for element in root.findall(".//tool_call"):
                json_data = self._parse_json(element.text.strip())
                if json_data:
                    tool_calls.append(json_data)
            
            return bool(tool_calls), tool_calls, None
            
        except ET.ParseError as e:
            error_msg = f"XML parsing error: {str(e)}"
            self.logger.error(error_msg)
            return False, [], error_msg
        
        except Exception as e:
            error_msg = f"Tool call extraction error: {str(e)}"
            self.logger.error(error_msg)
            return False, [], error_msg

    def extract_json_from_markdown(self, text: str) -> Optional[Dict]:
        """
        Extract JSON from markdown text.
        
        Args:
            text: Markdown text containing JSON
            
        Returns:
            Optional[Dict]: Extracted JSON data or None if extraction fails
        """
        try:
            # Try to find JSON in code blocks
            json_pattern = r'```json\r?\n(.*?)\r?\n```'
            match = re.search(json_pattern, text, re.DOTALL)
            
            if match:
                return self._parse_json(match.group(1))
            
            # Try to find JSON between curly braces
            brace_pattern = r'\{[\s\S]*\}'
            match = re.search(brace_pattern, text)
            
            if match:
                return self._parse_json(match.group(0))
            
            self.logger.warning("No JSON found in markdown text")
            return None
            
        except Exception as e:
            self.logger.error(f"Error extracting JSON from markdown: {e}")
            return None

    def _parse_json(self, text: str) -> Optional[Dict]:
        """
        Parse JSON text using multiple methods.
        
        Args:
            text: JSON text to parse
            
        Returns:
            Optional[Dict]: Parsed JSON data or None if parsing fails
        """
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(text)
            except (SyntaxError, ValueError):
                self.logger.error(f"Failed to parse JSON: {text}")
                return None

class Utils:
    """Utility functions for the application"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.default_ascii_text = "NousResearch  -  IceMasterTresearch"

    def print_ascii_art(self, text: str = None, font: str = "nancyj") -> None:
        """
        Print ASCII art text.
        If no text is provided, uses default NousResearch - IceMasterTresearch text.
        
        Args:
            text: Optional custom text to display
            font: Font to use for ASCII art
        """
        try:
            display_text = text or self.default_ascii_text
            ascii_art = text2art(display_text, font=font)
            print("\n" + ascii_art + "\n")
        except Exception as e:
            self.logger.error(f"Error generating ASCII art: {e}")
            print(f"\n{self.default_ascii_text}\n")  # Fallback to plain text

    def load_chat_template(self, template_name: str) -> Optional[str]:
        """
        Load chat template from file.
        
        Args:
            template_name: Name of the template to load
            
        Returns:
            Optional[str]: Template content or None if loading fails
        """
        try:
            template_path = os.path.join(
                self.script_dir, 
                'chat_templates', 
                f"{template_name}.j2"
            )
            
            if not os.path.exists(template_path):
                raise FileNotFoundError(f"Template not found: {template_name}")
                
            with open(template_path, 'r') as file:
                return file.read()
                
        except Exception as e:
            self.logger.error(f"Error loading chat template: {e}")
            return None

    def load_few_shot_examples(self, num_examples: int) -> List[Dict]:
        """
        Load few-shot examples from file.
        
        Args:
            num_examples: Number of examples to load
            
        Returns:
            List[Dict]: List of example dictionaries
        """
        try:
            example_path = os.path.join(
                self.script_dir,
                'prompt_assets',
                'few_shot.json'
            )
            
            with open(example_path, 'r') as file:
                examples = json.load(file)
                
            if num_examples > len(examples):
                raise ValueError(
                    f"Not enough examples (requested {num_examples}, "
                    f"but only {len(examples)} available)"
                )
                
            return examples[:num_examples]
            
        except Exception as e:
            self.logger.error(f"Error loading few-shot examples: {e}")
            return []

# Initialize global instances
logger = LoggerSetup.setup_logger("inference_logger")
text_processor = TextProcessor(logger)
utils = Utils(logger)

def main():
    """Main function to demonstrate functionality"""
    try:
        # Display ASCII art header
        utils.print_ascii_art()
        
        # Test text processing
        completion = "<|assistant|>Here's a test message</|assistant|>"
        message = text_processor.get_assistant_message(completion, "zephyr", "</s>")
        print(f"Extracted message: {message}")
        
        # Test JSON extraction
        markdown_text = "```json\n{\"key\": \"value\"}\n```"
        json_data = text_processor.extract_json_from_markdown(markdown_text)
        print(f"Extracted JSON: {json_data}")
        
        # Test template loading
        template = utils.load_chat_template("default")
        if template:
            print("Successfully loaded chat template")
        
        # Test few-shot examples
        examples = utils.load_few_shot_examples(2)
        if examples:
            print(f"Loaded {len(examples)} few-shot examples")
            
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise

if __name__ == "__main__":
    main()
