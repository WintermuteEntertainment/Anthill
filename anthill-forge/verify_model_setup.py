# verify_model_setup.py
import torch
from pathlib import Path
import subprocess
import sys

def check_network_drive():
    """Check if Z: drive is properly mounted"""
    print("🔌 Checking network drive Z:...")
    
    # Check if Z: exists
    z_drive = Path("Z:/")
    if not z_drive.exists():
        print("❌ Z: drive not found!")
        print("\n   Mount it with:")
        print("   net use Z: \\\\Bigblackbox\\a /persistent:yes")
        
        # Try to mount it automatically
        response = input("\n   Attempt to mount automatically? (yes/NO): ").strip().lower()
        if response == 'yes':
            print("   Mounting Z: drive...")
            result = subprocess.run(
                ['net', 'use', 'Z:', r'\\Bigblackbox\a', '/persistent:yes'],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                print("   ✅ Drive mounted successfully")
            else:
                print(f"   ❌ Failed to mount: {result.stderr}")
                return False
        else:
            return False
    else:
        print("✅ Z: drive is mounted")
    
    return True

def verify_model_structure():
    """Verify the model directory structure"""
    model_path = Path("Z:/gertrude_phi2_finetune/final_model")
    
    print(f"\n📁 Verifying model at: {model_path}")
    
    if not model_path.exists():
        print("❌ Model directory doesn't exist!")
        print(f"   Check: {model_path}")
        print(f"   Network path: \\\\Bigblackbox\\a\\gertrude_phi2_finetune\\final_model")
        return False
    
    print("✅ Model directory exists")
    
    # List all files
    files = list(model_path.iterdir())
    print(f"\n📄 Found {len(files)} files:")
    
    for file in files:
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"   • {file.name:30} {size_mb:6.1f} MB")
    
    # Check for critical files
    critical_files = ["config.json", "model.safetensors", "tokenizer_config.json"]
    missing = []
    
    print(f"\n🔍 Checking for critical files:")
    for cf in critical_files:
        cf_path = model_path / cf
        if cf_path.exists():
            print(f"   ✅ {cf}")
        else:
            # Check for variations
            found = False
            for f in files:
                if cf in f.name.lower():
                    print(f"   ⚠️  {cf} -> Found: {f.name}")
                    found = True
                    break
            if not found:
                print(f"   ❌ {cf}")
                missing.append(cf)
    
    if missing:
        print(f"\n❌ Missing critical files: {missing}")
        return False
    
    return True

def test_model_loading():
    """Test if model can be loaded"""
    print(f"\n🤖 Testing model loading...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model_path = "Z:/gertrude_phi2_finetune/final_model"
        
        print("   Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        print(f"   ✅ Tokenizer loaded (vocab: {tokenizer.vocab_size})")
        
        print("   Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True
        )
        print(f"   ✅ Model loaded to {model.device}")
        
        # Test a simple inference
        print("   Testing inference...")
        test_prompt = "Human: What is 2+2?\n\nAssistant:"
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=50,
                temperature=0.7,
                do_sample=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"   Test response: {response[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_gpu():
    """Check GPU availability"""
    print(f"\n🎮 Checking GPU...")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA is available")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"   GPU {i}: {gpu_name} ({memory:.1f} GB)")
        
        # Test GPU memory
        try:
            torch.cuda.empty_cache()
            test_tensor = torch.randn(1000, 1000, device='cuda')
            print(f"   GPU memory test: OK")
        except Exception as e:
            print(f"   ❌ GPU memory test failed: {e}")
    else:
        print("⚠️  No GPU found. Training will be VERY slow on CPU.")
        print("   Consider using a machine with NVIDIA GPU.")
    
    return True

def main():
    print("=" * 70)
    print("ANTHILL FORGE - SETUP VERIFICATION")
    print("=" * 70)
    print("This script verifies everything is ready for training.")
    print("=" * 70)
    
    all_ok = True
    
    # Check network drive
    if not check_network_drive():
        all_ok = False
    
    # Check model structure
    if all_ok and not verify_model_structure():
        all_ok = False
    
    # Check GPU
    check_gpu()  # Just info, not fatal
    
    # Test model loading (only if everything else is OK)
    if all_ok:
        if not test_model_loading():
            all_ok = False
    
    print("\n" + "=" * 70)
    if all_ok:
        print("✅ SETUP VERIFIED - Ready for training!")
        print("\n   Run: python train_instruction_model_strict.py")
    else:
        print("❌ SETUP FAILED - Fix issues above before training.")
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
