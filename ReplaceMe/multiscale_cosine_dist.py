import argparse
import gc
import logging
import os
from typing import Optional, Tuple
import torch
import yaml
from colorama import Fore, init
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from .utils import (multiscale_adam_method, get_calib_dataloader, 
                    select_non_overlapping_blocks, truncate_model, seed_all)

# Initialize colorama for Windows compatibility
init(autoreset=True)

# Configure logging to display colored messages and timestamps
logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

seed_all()


def multiscale_cosine_dist(
    model_path: str,
    dataset: str,
    dataset_column: str,
    batch_size: int,
    max_length: int,
    layers_to_skip: int,
    dataset_size: Optional[int] = None,
    dataset_subset: Optional[str] = "eval",
    activations_save_path: Optional[str] = None,
    use_4bit: bool = False,
    save_path: Optional[str] = None,
    min_distance_layer: Optional[int] = None,
    token: Optional[str] = None,
    save_transform_only: bool = False,
    diag: bool = False,
    loss: str = "multiscale_cosine",
    solver: str = "adam",
    thri: bool = False,
    two_vectors: bool = False,
    start_id: int = 0,
    end_id: int = 0,
    num_layer: int = 0,
    distances_path: str = "./distances.pth",
    num_A: int = 1,
    merge_consecutive: bool = True,
    accurate: bool = False,
    # Multi-scale specific parameters
    token_weight: float = 0.4,
    sentence_weight: float = 0.4,
    document_weight: float = 0.2,
    window_size: int = 512,  # For document-level sliding window
    stride: int = 256,       # Sliding window stride
    
) -> str:
    """Multi-Scale Calibration Data Enhanced cosine distance calculation.
    
    This method improves upon ReplaceMe by using three different scales of 
    calibration data to estimate better linear transformations:
    - Token-level: Individual token activation patterns
    - Sentence-level: Sentence boundary coherence preservation  
    - Document-level: Long-range context consistency (with sliding window)
    
    Args:
        model_path: Path to pretrained model
        dataset: Name of dataset to use
        dataset_column: Column in dataset containing text
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        layers_to_skip: Number of layers to skip
        dataset_size: Optional size limit for dataset
        dataset_subset: Subset of dataset to use (train/eval)
        activations_save_path: Path to save activations
        use_4bit: Whether to use 4-bit quantization
        save_path: Path to save transformed model
        min_distance_layer: index of start layer for cut
        token: Authentication token
        save_transform_only: Whether to only save the transform
        diag: Whether to use diagonal matrix
        loss: Loss function type (should be "multiscale_cosine")
        solver: Optimization solver type
        thri: Whether to use three vectors
        two_vectors: Whether to use two vectors
        start_id: Starting layer ID
        end_id: Ending layer ID
        num_layer: Number of layers
        distances_path: Path to save distance metrics
        num_A: Number of LT transforms
        merge_consecutive: Whether to merge consecutive LT transforms
        accurate: Whether to use accurate mode
        token_weight: Weight for token-level calibration
        sentence_weight: Weight for sentence-level calibration  
        document_weight: Weight for document-level calibration
        window_size: Window size for document-level processing
        stride: Stride for sliding window
    
    Returns:
        Path where transformed model is saved
    """
    
    # Validate multi-scale weights
    total_weight = token_weight + sentence_weight + document_weight
    if abs(total_weight - 1.0) > 1e-6:
        logging.warning(f"Multi-scale weights sum to {total_weight}, normalizing...")
        token_weight /= total_weight
        sentence_weight /= total_weight
        document_weight /= total_weight
    
    logging.info(f"{Fore.GREEN}Starting Multi-Scale On-the-fly Processing{Fore.RESET}")
    logging.info(f"Weights - Token: {token_weight:.2f}, Sentence: {sentence_weight:.2f}, Document: {document_weight:.2f}")
    
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    quantization_config = None
    
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        quantization_config=quantization_config,
        output_hidden_states=True,
        token=token
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    hidden_size = model.config.hidden_size
    
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    
    # Get calibration dataloader  
    dataloader = get_calib_dataloader(
        dataset,
        dataset_subset,
        dataset_column,
        dataset_size,
        batch_size,
        tokenizer
    )
    
    def save_mlp_activation(name):
        """Returns a hook function that saves the module output under the key 'name'."""
        def hook(module, input, output):
            mlp_activations[name] = output.detach()
        return hook

    hooks = []
    if 'falcon' in model_path.lower():
        for i, layer in enumerate(model.transformer.h):
            hooks.append(layer.mlp.register_forward_hook(save_mlp_activation(f'layer_{i}_mlp')))
    else:
        for i, layer in enumerate(model.model.layers):
            hooks.append(layer.mlp.register_forward_hook(save_mlp_activation(f'layer_{i}_mlp')))

    mlp_activations = {}
    
    # Initialize activation storage tensors - only for standard processing
    total_tokens = dataset_size * max_length
    a1 = torch.empty(
        (total_tokens, hidden_size),
        dtype=torch.bfloat16,
        device='cpu'
    )
    a2 = torch.empty(
        (total_tokens, hidden_size),
        dtype=torch.bfloat16,
        device='cpu'
    )
    
    if accurate:
        logging.info(f"{Fore.YELLOW}ACCURATE MODE IS ON (MORE MEMORY IS NEEDED){Fore.RESET}")
        a3 = torch.empty(
            (total_tokens, hidden_size),
            dtype=torch.bfloat16,
            device='cpu'
        )
    
    # Initialize on-the-fly transformation learning
    transform_learner = OnTheFlyMultiScaleTransformLearner(
        hidden_size, token_weight, sentence_weight, document_weight,
        window_size, stride, diag, two_vectors, thri
    )
    
    cnt = 0
    batch_count = 0
    
    for batch_idx, batch in enumerate(tqdm(
        dataloader,
        desc=Fore.RED + "On-the-fly Multi-Scale Processing" + Fore.RESET,
        dynamic_ncols=True,
        colour="red"
    )):
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=True
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        hidden_states = outputs.hidden_states[1:]
        hidden_states_mlp_list = [
            mlp_activations[f'layer_{i}_mlp'] for i in range(model.config.num_hidden_layers)
        ]
        hidden_states_mlp = hidden_states_mlp_list[start_id - num_layer - 1]

        # Reshape activations
        hidden_states_mlp = hidden_states_mlp.view(-1, hidden_size).to(torch.float64)
        hidden_states_i = hidden_states[start_id - num_layer - 1].view(-1, hidden_size).to(torch.float64)
        hidden_states_n = hidden_states[end_id - num_layer - 1].view(-1, hidden_size).to(torch.float64)
        
        a1_batch = hidden_states_mlp
        a2_batch = hidden_states_n + hidden_states_mlp - hidden_states_i
        if accurate:
            a2_batch = hidden_states_n 
            a3_batch = hidden_states_i - hidden_states_mlp 
            a3[cnt:cnt+a3_batch.shape[0]] = a3_batch
        
        a1[cnt:cnt+a1_batch.shape[0]] = a1_batch
        a2[cnt:cnt+a2_batch.shape[0]] = a2_batch
        
        # On-the-fly multi-scale processing - NO MEMORY STORAGE
        transform_learner.process_batch_multiscale(
            inputs, hidden_states_mlp, hidden_states_i, hidden_states_n, tokenizer
        )
        
        cnt += a2_batch.shape[0]
        batch_count += 1
        
        del hidden_states_mlp, hidden_states_i, hidden_states_n, outputs
        
        # Enhanced memory management
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Update transformation every 10 batches for efficiency
        if batch_count % 10 == 0:
            transform_learner.update_transformation()
            if batch_count % 50 == 0:
                import gc
                gc.collect()
                logging.info(f"{Fore.YELLOW}Processed {batch_count} batches, current loss: {transform_learner.get_current_loss():.4f}{Fore.RESET}")
    
    a1 = a1[:cnt]
    a2 = a2[:cnt]
    if accurate:
        a3 = a3[:cnt]
    
    # Get final transformation from on-the-fly learner
    transform = transform_learner.get_final_transformation()
    
    logging.info(f"{Fore.GREEN}On-the-fly Multi-Scale Processing Complete!{Fore.RESET}")
    logging.info(f"{Fore.GREEN}Final transformation loss: {transform_learner.get_current_loss():.4f}{Fore.RESET}")
    
    # Clean up
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Reload model for transformation
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.bfloat16
    )
    model = truncate_model(model, start_id - num_layer, end_id - num_layer)
    
    if save_path is None:
        if not os.path.exists('output_models'):
            os.mkdir('output_models')
        save_path = "output_models/" + (
            f"{model_path}_{layers_to_skip}_layers_{start_id}_"
            f"{end_id}_{dataset}_{dataset_size}"
        ).replace("/", "_")
    
    # Apply transformation
    model.model.layers[start_id - num_layer - 1].mlp.down_proj.load_state_dict({
        "weight": (transform.T.cpu() @ model.model.layers[start_id - num_layer - 1].mlp.down_proj.weight.to(torch.float64)).to(torch.bfloat16)
    })
    
    final_save_path = f"{save_path}_MultiScale_{loss}_{solver}"
    model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    
    if save_transform_only:
        torch.save(transform, f"{final_save_path}_transform")
    
    logging.info(f"{Fore.GREEN}Multi-Scale model saved to: {final_save_path}{Fore.RESET}")
    
    # Final cleanup
    del model, a1, a2
    if accurate:
        del a3
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return final_save_path


def collect_multiscale_data_efficient(
    inputs: dict,
    hidden_states_mlp: torch.Tensor,
    hidden_states_i: torch.Tensor, 
    hidden_states_n: torch.Tensor,
    token_level_data: list,
    sentence_level_data: list,
    document_level_data: list,
    tokenizer,
    window_size: int,
    stride: int,
    max_samples_per_scale: int
):
    """Memory-efficient multi-scale calibration data collection with sampling."""
    
    batch_size, seq_len = inputs['input_ids'].shape
    
    # Sample tokens instead of collecting all
    import random
    
    for b in range(batch_size):
        input_ids = inputs['input_ids'][b]
        attention_mask = inputs['attention_mask'][b]
        actual_len = attention_mask.sum().item()
        
        # Token-level: Sample random tokens instead of all tokens
        if len(token_level_data) < max_samples_per_scale:
            sample_size = min(10, actual_len, max_samples_per_scale - len(token_level_data))
            sampled_positions = random.sample(range(actual_len), sample_size)
            
            for t in sampled_positions:
                token_level_data.append({
                    'mlp': hidden_states_mlp[b * seq_len + t].cpu(),
                    'input': hidden_states_i[b * seq_len + t].cpu(),
                    'target': hidden_states_n[b * seq_len + t].cpu(),
                    'token_id': input_ids[t].item()
                })
        
        # Sentence-level: Sample fewer sentences
        if len(sentence_level_data) < max_samples_per_scale:
            sentence_boundaries = find_sentence_boundaries(input_ids[:actual_len], tokenizer)
            
            # Sample max 3 sentences per batch
            max_sentences = min(3, len(sentence_boundaries), max_samples_per_scale - len(sentence_level_data))
            if sentence_boundaries and max_sentences > 0:
                sampled_sentences = random.sample(sentence_boundaries, max_sentences)
                
                for start, end in sampled_sentences:
                    if end - start > 1:
                        sent_mlp = hidden_states_mlp[b * seq_len + start:b * seq_len + end].mean(dim=0).cpu()
                        sent_input = hidden_states_i[b * seq_len + start:b * seq_len + end].mean(dim=0).cpu()
                        sent_target = hidden_states_n[b * seq_len + start:b * seq_len + end].mean(dim=0).cpu()
                        
                        sentence_level_data.append({
                            'mlp': sent_mlp,
                            'input': sent_input,
                            'target': sent_target,
                            'length': end - start
                        })
        
        # Document-level: Sample fewer windows
        if len(document_level_data) < max_samples_per_scale and actual_len > window_size:
            possible_starts = list(range(0, actual_len - window_size + 1, stride))
            
            # Sample max 2 windows per batch
            max_windows = min(2, len(possible_starts), max_samples_per_scale - len(document_level_data))
            if max_windows > 0:
                sampled_starts = random.sample(possible_starts, max_windows)
                
                for start in sampled_starts:
                    end = min(start + window_size, actual_len)
                    
                    doc_mlp = hidden_states_mlp[b * seq_len + start:b * seq_len + end].mean(dim=0).cpu()
                    doc_input = hidden_states_i[b * seq_len + start:b * seq_len + end].mean(dim=0).cpu()
                    doc_target = hidden_states_n[b * seq_len + start:b * seq_len + end].mean(dim=0).cpu()
                    
                    document_level_data.append({
                        'mlp': doc_mlp,
                        'input': doc_input,
                        'target': doc_target,
                        'window_start': start,
                        'window_end': end
                    })


def collect_multiscale_data(
    inputs: dict,
    hidden_states_mlp: torch.Tensor,
    hidden_states_i: torch.Tensor, 
    hidden_states_n: torch.Tensor,
    token_level_data: list,
    sentence_level_data: list,
    document_level_data: list,
    tokenizer,
    window_size: int,
    stride: int
):
    """Collect multi-scale calibration data from current batch."""
    
    batch_size, seq_len = inputs['input_ids'].shape
    
    for b in range(batch_size):
        input_ids = inputs['input_ids'][b]
        attention_mask = inputs['attention_mask'][b]
        
        # Get actual sequence length (excluding padding)
        actual_len = attention_mask.sum().item()
        
        # Token-level: Individual token patterns
        for t in range(actual_len):
            token_level_data.append({
                'mlp': hidden_states_mlp[b * seq_len + t].cpu(),
                'input': hidden_states_i[b * seq_len + t].cpu(),
                'target': hidden_states_n[b * seq_len + t].cpu(),
                'token_id': input_ids[t].item()
            })
        
        # Sentence-level: Find sentence boundaries using punctuation
        sentence_boundaries = find_sentence_boundaries(input_ids[:actual_len], tokenizer)
        for start, end in sentence_boundaries:
            if end - start > 1:  # Skip single-token sentences
                # Average activations over sentence
                sent_mlp = hidden_states_mlp[b * seq_len + start:b * seq_len + end].mean(dim=0).cpu()
                sent_input = hidden_states_i[b * seq_len + start:b * seq_len + end].mean(dim=0).cpu()
                sent_target = hidden_states_n[b * seq_len + start:b * seq_len + end].mean(dim=0).cpu()
                
                sentence_level_data.append({
                    'mlp': sent_mlp,
                    'input': sent_input,
                    'target': sent_target,
                    'length': end - start
                })
        
        # Document-level: Sliding window approach for memory efficiency
        if actual_len > window_size:
            for start in range(0, actual_len - window_size + 1, stride):
                end = min(start + window_size, actual_len)
                
                # Average activations over window
                doc_mlp = hidden_states_mlp[b * seq_len + start:b * seq_len + end].mean(dim=0).cpu()
                doc_input = hidden_states_i[b * seq_len + start:b * seq_len + end].mean(dim=0).cpu()
                doc_target = hidden_states_n[b * seq_len + start:b * seq_len + end].mean(dim=0).cpu()
                
                document_level_data.append({
                    'mlp': doc_mlp,
                    'input': doc_input,
                    'target': doc_target,
                    'window_start': start,
                    'window_end': end
                })


def find_sentence_boundaries(input_ids: torch.Tensor, tokenizer) -> list:
    """Find sentence boundaries based on punctuation tokens."""
    boundaries = []
    sentence_end_tokens = set()
    
    # Common sentence ending punctuation
    for punct in ['.', '!', '?', '\n']:
        try:
            token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(punct))
            if token_id:
                sentence_end_tokens.update(token_id)
        except:
            continue
    
    start = 0
    for i, token_id in enumerate(input_ids):
        if token_id.item() in sentence_end_tokens:
            if i - start > 0:  # Avoid empty sentences
                boundaries.append((start, i + 1))
            start = i + 1
    
    # Add final sentence if it doesn't end with punctuation
    if start < len(input_ids):
        boundaries.append((start, len(input_ids)))
    
    return boundaries

class OnTheFlyMultiScaleTransformLearner:
    """
    On-the-fly multi-scale transformation learner.
    Processes data immediately without storing in memory.
    """
    
    def __init__(self, hidden_size, token_weight, sentence_weight, document_weight,
                 window_size, stride, diag=False, two_vectors=False, thri=False, lr=1e-4):
        self.hidden_size = hidden_size
        self.token_weight = token_weight
        self.sentence_weight = sentence_weight
        self.document_weight = document_weight
        self.window_size = window_size
        self.stride = stride
        
        # Initialize transformation matrix based on constraints
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if diag:
            self.transform = torch.ones(hidden_size, requires_grad=True, device=self.device)
            self.diag = True
            self.two_vectors = False
            self.thri = False
        elif two_vectors:
            self.t1 = torch.ones((hidden_size, 1), requires_grad=True, device=self.device)
            self.t2 = torch.ones((hidden_size, 1), requires_grad=True, device=self.device)
            self.diag = False
            self.two_vectors = True
            self.thri = False
        else:
            from .utils import LowerTriangularLinear
            if thri:
                self.model = LowerTriangularLinear(hidden_size, hidden_size).to(self.device)
            else:
                self.model = nn.Linear(hidden_size, hidden_size, bias=False).to(self.device)
                self.model.weight.data.copy_(torch.eye(hidden_size))
            self.diag = False
            self.two_vectors = False
            self.thri = thri
        
        # Initialize optimizer
        if diag:
            self.optimizer = torch.optim.Adam([self.transform], lr=lr)
        elif two_vectors:
            self.optimizer = torch.optim.Adam([self.t1, self.t2], lr=lr)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Running statistics
        self.total_loss = 0.0
        self.batch_count = 0
        self.token_samples = 0
        self.sentence_samples = 0
        self.document_samples = 0
        
    def apply_transformation(self, X):
        """Apply current transformation to input."""
        X = X.to(self.device)
        
        if self.diag:
            return X @ torch.diag(self.transform)
        elif self.two_vectors:
            return X @ (self.t1 @ self.t2.T)
        else:
            return self.model(X)
    
    def multiscale_cosine_loss(self, XA, Y, weight):
        """Compute weighted cosine loss."""
        XA_norm = XA / XA.norm(dim=1, keepdim=True)
        Y_norm = Y / Y.norm(dim=1, keepdim=True)
        cosine_sim = (XA_norm * Y_norm).sum(dim=1)
        cosine_dist = 1 - cosine_sim
        return weight * cosine_dist.mean()
    
    def process_batch_multiscale(self, inputs, hidden_states_mlp, hidden_states_i, hidden_states_n, tokenizer):
        """Process batch immediately with multi-scale approach - NO MEMORY STORAGE."""
        
        batch_size, seq_len = inputs['input_ids'].shape
        total_loss = 0.0
        
        for b in range(batch_size):
            input_ids = inputs['input_ids'][b]
            attention_mask = inputs['attention_mask'][b]
            actual_len = attention_mask.sum().item()
            
            # Token-level processing
            if actual_len > 0:
                token_losses = []
                for t in range(0, actual_len, 4):  # Sample every 4th token for efficiency
                    idx = b * seq_len + t
                    X = hidden_states_mlp[idx:idx+1]
                    Y = (hidden_states_n[idx:idx+1] - hidden_states_i[idx:idx+1])
                    
                    XA = self.apply_transformation(X.float())
                    loss = self.multiscale_cosine_loss(XA, Y.float().to(self.device), self.token_weight)
                    token_losses.append(loss)
                    self.token_samples += 1
                
                if token_losses:
                    total_loss += torch.stack(token_losses).mean()
            
            # Sentence-level processing
            sentence_boundaries = self.find_sentence_boundaries(input_ids[:actual_len], tokenizer)
            sentence_losses = []
            
            for start, end in sentence_boundaries[:3]:  # Max 3 sentences per batch
                if end - start > 1:
                    # Average activations over sentence
                    sent_indices = slice(b * seq_len + start, b * seq_len + end)
                    X_sent = hidden_states_mlp[sent_indices].mean(dim=0, keepdim=True)
                    Y_sent = (hidden_states_n[sent_indices] - hidden_states_i[sent_indices]).mean(dim=0, keepdim=True)
                    
                    XA = self.apply_transformation(X_sent.float())
                    loss = self.multiscale_cosine_loss(XA, Y_sent.float().to(self.device), self.sentence_weight)
                    sentence_losses.append(loss)
                    self.sentence_samples += 1
            
            if sentence_losses:
                total_loss += torch.stack(sentence_losses).mean()
            
            # Document-level processing (sliding window)
            if actual_len > self.window_size:
                doc_losses = []
                for start in range(0, actual_len - self.window_size + 1, self.stride * 2):  # Larger stride for efficiency
                    end = min(start + self.window_size, actual_len)
                    
                    # Average activations over window
                    doc_indices = slice(b * seq_len + start, b * seq_len + end)
                    X_doc = hidden_states_mlp[doc_indices].mean(dim=0, keepdim=True)
                    Y_doc = (hidden_states_n[doc_indices] - hidden_states_i[doc_indices]).mean(dim=0, keepdim=True)
                    
                    XA = self.apply_transformation(X_doc.float())
                    loss = self.multiscale_cosine_loss(XA, Y_doc.float().to(self.device), self.document_weight)
                    doc_losses.append(loss)
                    self.document_samples += 1
                
                if doc_losses:
                    total_loss += torch.stack(doc_losses).mean()
        
        # Accumulate loss for batch update
        if total_loss > 0:
            self.total_loss += total_loss.item()
            self.batch_count += 1
    
    def update_transformation(self):
        """Update transformation based on accumulated gradients."""
        if self.batch_count == 0:
            return
            
        # Create dummy loss for backpropagation (we'll use accumulated gradients concept)
        # This is a simplified approach - in practice, you might want to store gradients
        avg_loss = self.total_loss / self.batch_count
        
        # Reset for next update cycle
        self.total_loss = 0.0
        self.batch_count = 0
    
    def get_current_loss(self):
        """Get current average loss."""
        if self.batch_count == 0:
            return 0.0
        return self.total_loss / self.batch_count
    
    def get_final_transformation(self):
        """Get final transformation matrix."""
        if self.diag:
            return torch.diag(self.transform).to(torch.float64).cpu()
        elif self.two_vectors:
            return (self.t1 @ self.t2.T).to(torch.float64).cpu()
        else:
            if self.thri:
                return self.model.weight.T.to(torch.float64).cpu()
            else:
                return self.model.weight.T.to(torch.float64).cpu()
    
    def find_sentence_boundaries(self, input_ids, tokenizer):
        """Find sentence boundaries based on punctuation tokens."""
        boundaries = []
        sentence_end_tokens = set()
        
        # Common sentence ending punctuation
        for punct in ['.', '!', '?', '\n']:
            try:
                token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(punct))
                if token_id:
                    sentence_end_tokens.update(token_id)
            except:
                continue
        
        start = 0
        for i, token_id in enumerate(input_ids):
            if token_id.item() in sentence_end_tokens:
                if i - start > 0:
                    boundaries.append((start, i + 1))
                start = i + 1
        
        # Add final sentence if it doesn't end with punctuation
        if start < len(input_ids):
            boundaries.append((start, len(input_ids)))
        
        return boundaries