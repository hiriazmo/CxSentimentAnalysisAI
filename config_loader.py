"""
Configuration Loader
Loads settings from config.yaml for agent personas and prompts
"""

import yaml
import os
from typing import Dict, Any

class Config:
    """
    Configuration manager for the Review Intelligence System
    Loads and provides access to config.yaml settings
    """
    
    def __init__(self, config_file: str = "config.yaml"):
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_file):
            print(f"⚠️  Config file not found: {self.config_file}")
            print("   Using default configuration")
            return self._default_config()
        
        try:
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✅ Configuration loaded from {self.config_file}")
            return config
        except Exception as e:
            print(f"⚠️  Error loading config: {e}")
            print("   Using default configuration")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration if YAML not available"""
        return {
            'models': {
                'stage1': {
                    'llm1': {'name': 'Qwen/Qwen2.5-72B-Instruct', 'temperature': 0.1},
                    'llm2': {'name': 'mistralai/Mistral-7B-Instruct-v0.3', 'temperature': 0.1},
                    'manager': {'name': 'meta-llama/Llama-3.1-8B-Instruct', 'temperature': 0.1}
                },
                'stage2': {
                    'best_model': {'name': 'cardiffnlp/twitter-roberta-base-sentiment-latest'},
                    'alternate_model': {'name': 'finiteautomata/bertweet-base-sentiment-analysis'}
                },
                'stage3': {
                    'llm3': {'name': 'meta-llama/Llama-3.1-70B-Instruct', 'temperature': 0.1}
                }
            }
        }
    
    def get_model(self, stage: str, model_key: str) -> Dict[str, Any]:
        """Get model configuration for a specific stage"""
        return self.config.get('models', {}).get(stage, {}).get(model_key, {})
    
    def get_persona(self, agent: str) -> Dict[str, Any]:
        """Get persona configuration for an agent"""
        return self.config.get('personas', {}).get(agent, {})
    
    def get_prompt_template(self, template_name: str) -> str:
        """Get prompt template"""
        return self.config.get('prompt_templates', {}).get(template_name, '')
    
    def get_classification_rules(self) -> Dict[str, Any]:
        """Get classification rules"""
        return self.config.get('classification_rules', {})
    
    def get_sentiment_settings(self) -> Dict[str, Any]:
        """Get sentiment analysis settings"""
        return self.config.get('sentiment', {})
    
    def get_batch_settings(self) -> Dict[str, Any]:
        """Get batch analysis settings"""
        return self.config.get('batch_analysis', {})
    
    def get_processing_settings(self) -> Dict[str, Any]:
        """Get processing settings"""
        return self.config.get('processing', {})
    
    def get_dashboard_settings(self) -> Dict[str, Any]:
        """Get dashboard settings"""
        return self.config.get('dashboard', {})


# Singleton instance
_config_instance = None

def get_config(config_file: str = "config.yaml") -> Config:
    """Get or create config singleton"""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_file)
    return _config_instance


if __name__ == "__main__":
    # Test config loader
    print("\n" + "="*60)
    print("🧪 TESTING CONFIG LOADER")
    print("="*60 + "\n")
    
    config = get_config()
    
    # Test model access
    llm1_config = config.get_model('stage1', 'llm1')
    print(f"LLM1 Model: {llm1_config.get('name', 'Not found')}")
    
    # Test persona access
    llm1_persona = config.get_persona('llm1')
    print(f"LLM1 Persona: {llm1_persona.get('name', 'Not found')}")
    
    # Test prompt template
    prompt = config.get_prompt_template('stage1_llm1')
    print(f"Prompt template loaded: {len(prompt)} characters")
    
    print("\n✅ Config loader test complete!")
