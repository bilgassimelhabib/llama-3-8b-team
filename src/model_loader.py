from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from config import config
import time

class ModelLoader:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        
    def load_model(self, quantize_4bit=True):
        """Charge le modèle avec optimisation mémoire"""
        
        if self.is_loaded:
            return self.model, self.tokenizer
            
        print(f"🔄 Chargement du modèle {config.MODEL_NAME} sur {config.device}...")
        start_time = time.time()
        
        # Configuration de quantization pour économiser la RAM
        if quantize_4bit and config.device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
        else:
            bnb_config = None
            
        # Chargement du tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.MODEL_NAME,
            cache_dir=config.model_cache,
            trust_remote_code=True
        )
        
        # Ajout du pad token si nécessaire
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Chargement du modèle
        self.model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto" if config.device == "cuda" else None,
            torch_dtype=torch.float16 if config.device == "cuda" else torch.float32,
            cache_dir=config.model_cache,
            trust_remote_code=True
        )
        
        self.is_loaded = True
        load_time = time.time() - start_time
        print(f"✅ Modèle chargé en {load_time:.2f} secondes")
        
        return self.model, self.tokenizer

    def generate_text(self, prompt, max_length=512, temperature=0.7, top_p=0.9):
        """Génère du texte à partir d'un prompt"""
        if not self.is_loaded:
            self.load_model()
            
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
