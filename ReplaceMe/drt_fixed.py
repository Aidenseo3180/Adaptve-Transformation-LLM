import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from typing import Dict, List, Tuple

class DRTWrapper(nn.Module):
    """실제로 작동하는 DRT Wrapper"""
    
    def __init__(self, base_model, drt_config):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.drt_config = drt_config
        
        # DRT 설정
        self.merge_threshold = drt_config['merge_threshold']
        self.start_merge_layer = drt_config['start_merge_layer']
        self.layer_merge_stats = drt_config.get('layer_merge_stats', {})
        
        # 실제로 forward hook 적용
        self._apply_drt_hooks()
        
        print(f"[DRT] Applied hooks to layers {self.start_merge_layer} to {self.config.num_hidden_layers-1}")
    
    def _apply_drt_hooks(self):
        """실제로 작동하는 hook 적용"""
        
        def create_forward_hook(layer_idx):
            def hook(module, input, output):
                hidden_states = output[0] if isinstance(output, tuple) else output
                
                # 깊이 비율 계산
                depth_ratio = (layer_idx - self.start_merge_layer) / (self.config.num_hidden_layers - self.start_merge_layer)
                adaptive_threshold = self.merge_threshold - (0.2 * depth_ratio)
                
                # 토큰 병합 로직
                batch_size, seq_len, hidden_dim = hidden_states.shape
                
                if seq_len > 10:  # 너무 짧은 시퀀스는 병합 안 함
                    # 병합 수 계산 (통계 기반)
                    if str(layer_idx) in self.layer_merge_stats:
                        avg_merges = sum(self.layer_merge_stats[str(layer_idx)]) / len(self.layer_merge_stats[str(layer_idx)])
                        merge_ratio = min(avg_merges * 2 / seq_len, 0.6)  # 최대 60% 병합
                    else:
                        merge_ratio = 0.2 * depth_ratio  # 기본값
                    
                    # 실제 병합 수행
                    num_tokens_to_keep = int(seq_len * (1 - merge_ratio))
                    num_tokens_to_keep = max(num_tokens_to_keep, seq_len // 3)  # 최소 1/3 유지
                    
                    if num_tokens_to_keep < seq_len:
                        # Token importance scoring (간단한 norm 기반)
                        token_scores = hidden_states.norm(dim=-1)  # [batch, seq_len]
                        
                        # Keep top-k tokens
                        _, keep_indices = token_scores.topk(num_tokens_to_keep, dim=1)
                        keep_indices = keep_indices.sort(dim=1)[0]  # 순서 유지
                        
                        # 선택된 토큰만 유지
                        merged_hidden = torch.gather(
                            hidden_states, 
                            1, 
                            keep_indices.unsqueeze(-1).expand(-1, -1, hidden_dim)
                        )
                        
                        # Output 교체
                        if isinstance(output, tuple):
                            output = (merged_hidden,) + output[1:]
                        else:
                            output = merged_hidden
                        
                        # 디버그 출력 (처음 몇 번만)
                        if not hasattr(module, '_drt_logged'):
                            print(f"[DRT] Layer {layer_idx}: {seq_len} → {num_tokens_to_keep} tokens")
                            module._drt_logged = True
                
                return output
            
            return hook
        
        # 각 레이어에 hook 등록
        if hasattr(self.base_model, 'model'):
            layers = self.base_model.model.layers
        else:
            layers = self.base_model.transformer.h
        
        for idx in range(self.start_merge_layer, len(layers)):
            layers[idx].register_forward_hook(create_forward_hook(idx))
    
    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)
    
    @property
    def device(self):
        return self.base_model.device
    
    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)


def apply_drt_and_save(
    model_path: str,
    save_path: str,
    drt_config: Dict
):
    """모델에 DRT를 적용하고 저장"""
    
    print(f"[DRT] Loading model from {model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    print(f"[DRT] Creating DRT wrapper")
    drt_model = DRTWrapper(base_model, drt_config)
    
    # 테스트 forward pass
    print(f"[DRT] Testing forward pass...")
    dummy_input = torch.randint(0, 1000, (1, 50)).to(drt_model.device)
    with torch.no_grad():
        output = drt_model(dummy_input)
    print(f"[DRT] Test passed! Output shape: {output.logits.shape}")
    
    # 저장 (wrapper 전체를 저장)
    print(f"[DRT] Saving to {save_path}")
    torch.save({
        'model_state_dict': drt_model.state_dict(),
        'drt_config': drt_config,
        'base_model_config': base_model.config
    }, f"{save_path}/drt_model.pt")
    
    # Tokenizer도 저장
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(save_path)
    
    print(f"[DRT] Model saved successfully!")
    
    return drt_model