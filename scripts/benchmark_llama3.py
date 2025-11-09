#!/usr/bin/env python3
import time
import torch
import psutil
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class Llama3Benchmark:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.results = {}
        
    def print_system_info(self):
        print("🖥️ INFORMATIONS SYSTÈME")
        print("=" * 40)
        print(f"✅ CPU: {torch.get_num_threads()} cores")
        print(f"✅ RAM: {psutil.virtual_memory().total / 1e9:.1f} GB")
        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            print(f"✅ GPU: {gpu_props.name}")
            print(f"✅ VRAM: {gpu_props.total_memory / 1e9:.1f} GB")
        else:
            print("❌ Aucun GPU détecté")
        print()
    
    def load_model_4bit(self):
        print("🔄 CHARGEMENT MODÈLE 4-BIT")
        print("=" * 35)
        
        model_id = "meta-llama/Meta-Llama-3-8B"
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        start_time = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        load_time = time.time() - start_time
        print(f"✅ Modèle chargé en {load_time:.2f} secondes")
        print(f"📍 Device: {next(self.model.parameters()).device}")
        print(f"💾 Mémoire utilisée: {torch.cuda.memory_allocated() / 1e9:.1f} GB\n")
        
        self.results['load_time'] = load_time
        self.results['gpu_memory'] = torch.cuda.memory_allocated() / 1e9
    
    def run_benchmarks(self):
        print("📊 LANCEMENT DES BENCHMARKS")
        print("=" * 35)
        
        test_cases = [
            {"name": "Court (50 tokens)", "prompt": "Explique la physique quantique:", "max_length": 50},
            {"name": "Moyen (100 tokens)", "prompt": "Donne-moi 5 avantages de l'IA générative:", "max_length": 100},
            {"name": "Long (200 tokens)", "prompt": "Rédige un paragraphe sur l'impact de l'IA:", "max_length": 200}
        ]
        
        benchmarks = []
        
        for i, test in enumerate(test_cases):
            print(f"\n🧪 Test {i+1}: {test['name']}")
            print(f"   Prompt: {test['prompt']}")
            
            start_time = time.time()
            inputs = self.tokenizer(test['prompt'], return_tensors="pt")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=test['max_length'],
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generation_time = time.time() - start_time
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            input_tokens = len(inputs['input_ids'][0])
            output_tokens = len(outputs[0])
            new_tokens = output_tokens - input_tokens
            tokens_per_second = new_tokens / generation_time
            
            print(f"   ⏱️  Temps: {generation_time:.2f}s")
            print(f"   📝 Tokens générés: {new_tokens}")
            print(f"   🚀 Tokens/sec: {tokens_per_second:.1f}")
            print(f"   📄 Réponse: {generated_text[:80]}...")
            
            benchmarks.append({
                'name': test['name'],
                'time': generation_time,
                'tokens_generated': new_tokens,
                'tokens_per_second': tokens_per_second
            })
        
        self.results['benchmarks'] = benchmarks
        return benchmarks

def main():
    benchmark = Llama3Benchmark()
    benchmark.print_system_info()
    benchmark.load_model_4bit()
    benchmark.run_benchmarks()
    print("\n🎉 BENCHMARK TERMINÉ!")

if __name__ == "__main__":
    main()
