import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)
from .drt_fixed import DRTWrapper

def load_drt_model(save_path: str):
    """저장된 DRT 모델 로드"""
    
    # 체크포인트 로드
    checkpoint = torch.load(f"{save_path}/drt_model.pt", map_location="cpu")
    
    # Base model 로드
    base_config = checkpoint['base_model_config']
    base_model = AutoModelForCausalLM.from_config(base_config)
    
    # DRT wrapper 재생성
    drt_model = DRTWrapper(base_model, checkpoint['drt_config'])
    drt_model.load_state_dict(checkpoint['model_state_dict'])
    
    return drt_model


def test_drt_inference(model_path: str):
    """DRT 모델 추론 테스트"""
        
    model = load_drt_model(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 테스트 생성
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    print(f"[Test] Generating from: {prompt}")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"[Test] Response: {response}")