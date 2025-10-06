# ============================================================
# vlm_evaluator.py (새 파일 생성)
# ============================================================

"""VLM Evaluation Module for Multimodal Benchmarks"""

import argparse
import json
import logging
import os
from typing import Dict, List, Optional
from pathlib import Path

import torch
import yaml
from colorama import Fore, init
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset
from transformers import LlavaForConditionalGeneration

from .utils import setup_vlm_processor, seed_all

init(autoreset=True)

logging.basicConfig(
    format=(
        f"{Fore.CYAN}%(asctime)s "
        f"{Fore.YELLOW}[%(levelname)s] "
        f"{Fore.RESET}%(message)s"
    ),
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

seed_all()


def eval_vlm_vqav2(
    model,
    processor,
    num_samples: int = 1000,
    batch_size: int = 4
) -> Dict:
    """Evaluate on VQAv2 dataset.
    
    Args:
        model: VLM model
        processor: VLM processor
        num_samples: Number of samples to evaluate
        batch_size: Batch size
        
    Returns:
        Evaluation results
    """
    print(f"{Fore.CYAN}Evaluating on VQAv2...{Fore.RESET}")
    
    try:
        # VQAv2 validation set 로드
        dataset = load_dataset(
            "HuggingFaceM4/VQAv2",
            split="validation"
        ).select(range(num_samples))
        
        correct = 0
        total = 0
        
        for i in tqdm(range(0, len(dataset), batch_size), desc="VQAv2"):
            batch = dataset[i:min(i+batch_size, len(dataset))]
            
            images = [item['image'].convert('RGB') for item in batch]
            questions = [item['question'] for item in batch]
            answers = [item['multiple_choice_answer'] for item in batch]
            
            # Processor로 입력 생성
            inputs = processor(
                text=questions,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=False
                )
            
            predictions = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )
            
            # 정확도 계산 (단순 문자열 매칭)
            for pred, ans in zip(predictions, answers):
                if ans.lower() in pred.lower():
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0
        print(f"{Fore.GREEN}VQAv2 Accuracy: {accuracy:.4f} ({correct}/{total}){Fore.RESET}")
        
        return {
            "vqav2_accuracy": accuracy,
            "vqav2_correct": correct,
            "vqav2_total": total
        }
        
    except Exception as e:
        print(f"{Fore.RED}VQAv2 evaluation failed: {e}{Fore.RESET}")
        return {"vqav2_accuracy": 0.0, "error": str(e)}


def eval_vlm_gqa(
    model,
    processor,
    num_samples: int = 1000,
    batch_size: int = 4
) -> Dict:
    """Evaluate on GQA dataset.
    
    Args:
        model: VLM model
        processor: VLM processor
        num_samples: Number of samples
        batch_size: Batch size
        
    Returns:
        Evaluation results
    """
    print(f"{Fore.CYAN}Evaluating on GQA...{Fore.RESET}")
    
    try:
        # GQA testdev 로드
        dataset = load_dataset(
            "Multimodal-Fatima/GQA_dev_balanced",
            split="train"
        ).select(range(num_samples))
        
        correct = 0
        total = 0
        
        for i in tqdm(range(0, len(dataset), batch_size), desc="GQA"):
            batch = dataset[i:min(i+batch_size, len(dataset))]
            
            images = [item['image'].convert('RGB') for item in batch]
            questions = [item['question'] for item in batch]
            answers = [item['answer'] for item in batch]
            
            inputs = processor(
                text=questions,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=False
                )
            
            predictions = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )
            
            for pred, ans in zip(predictions, answers):
                if ans.lower() in pred.lower():
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0
        print(f"{Fore.GREEN}GQA Accuracy: {accuracy:.4f} ({correct}/{total}){Fore.RESET}")
        
        return {
            "gqa_accuracy": accuracy,
            "gqa_correct": correct,
            "gqa_total": total
        }
        
    except Exception as e:
        print(f"{Fore.RED}GQA evaluation failed: {e}{Fore.RESET}")
        return {"gqa_accuracy": 0.0, "error": str(e)}


def eval_vlm_textvqa(
    model,
    processor,
    num_samples: int = 500,
    batch_size: int = 4
) -> Dict:
    """Evaluate on TextVQA dataset.
    
    Args:
        model: VLM model
        processor: VLM processor
        num_samples: Number of samples
        batch_size: Batch size
        
    Returns:
        Evaluation results
    """
    print(f"{Fore.CYAN}Evaluating on TextVQA...{Fore.RESET}")
    
    try:
        dataset = load_dataset(
            "facebook/textvqa",
            split="validation"
        ).select(range(num_samples))
        
        correct = 0
        total = 0
        
        for i in tqdm(range(0, len(dataset), batch_size), desc="TextVQA"):
            batch = dataset[i:min(i+batch_size, len(dataset))]
            
            images = [item['image'].convert('RGB') for item in batch]
            questions = [item['question'] for item in batch]
            answers = [item['answers'][0] for item in batch]  # 첫 번째 정답 사용
            
            inputs = processor(
                text=questions,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    do_sample=False
                )
            
            predictions = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )
            
            for pred, ans in zip(predictions, answers):
                if ans.lower() in pred.lower():
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0
        print(f"{Fore.GREEN}TextVQA Accuracy: {accuracy:.4f} ({correct}/{total}){Fore.RESET}")
        
        return {
            "textvqa_accuracy": accuracy,
            "textvqa_correct": correct,
            "textvqa_total": total
        }
        
    except Exception as e:
        print(f"{Fore.RED}TextVQA evaluation failed: {e}{Fore.RESET}")
        return {"textvqa_accuracy": 0.0, "error": str(e)}


def eval_vlm_ok_vqa(
    model,
    processor,
    num_samples: int = 500,
    batch_size: int = 4
) -> Dict:
    """Evaluate on OK-VQA dataset.
    
    Args:
        model: VLM model
        processor: VLM processor
        num_samples: Number of samples
        batch_size: Batch size
        
    Returns:
        Evaluation results
    """
    print(f"{Fore.CYAN}Evaluating on OK-VQA...{Fore.RESET}")
    
    try:
        dataset = load_dataset(
            "Multimodal-Fatima/OK-VQA_train",
            split="train"
        ).select(range(num_samples))
        
        correct = 0
        total = 0
        
        for i in tqdm(range(0, len(dataset), batch_size), desc="OK-VQA"):
            batch = dataset[i:min(i+batch_size, len(dataset))]
            
            images = [item['image'].convert('RGB') for item in batch]
            questions = [item['question'] for item in batch]
            answers = [item['answer'] for item in batch]
            
            inputs = processor(
                text=questions,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=False
                )
            
            predictions = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )
            
            for pred, ans in zip(predictions, answers):
                if ans.lower() in pred.lower():
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0
        print(f"{Fore.GREEN}OK-VQA Accuracy: {accuracy:.4f} ({correct}/{total}){Fore.RESET}")
        
        return {
            "okvqa_accuracy": accuracy,
            "okvqa_correct": correct,
            "okvqa_total": total
        }
        
    except Exception as e:
        print(f"{Fore.RED}OK-VQA evaluation failed: {e}{Fore.RESET}")
        return {"okvqa_accuracy": 0.0, "error": str(e)}


def vlm_evaluator(
    model_path: str,
    tasks: List[str] = None,
    num_samples: int = 1000,
    batch_size: int = 4,
    token: Optional[str] = None,
    **kwargs
) -> Dict:
    """Evaluate VLM on multimodal benchmarks.
    
    Args:
        model_path: Path to VLM model
        tasks: List of tasks to evaluate (default: all)
        num_samples: Number of samples per task
        batch_size: Batch size for evaluation
        token: HuggingFace token
        **kwargs: Additional arguments
        
    Returns:
        Dictionary of evaluation results
    """
    print(f"{Fore.GREEN}=== VLM Evaluation ==={Fore.RESET}")
    print(f"{Fore.CYAN}Model: {model_path}{Fore.RESET}")
    
    # 기본 tasks
    if tasks is None:
        tasks = ["vqav2", "gqa", "textvqa", "okvqa"]
    
    print(f"{Fore.CYAN}Tasks: {tasks}{Fore.RESET}")
    
    # 모델 로드
    print(f"{Fore.CYAN}Loading VLM model...{Fore.RESET}")
    
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=token
    )
    
    processor = setup_vlm_processor(model_path)
    model.eval()
    
    print(f"{Fore.GREEN}Model loaded successfully{Fore.RESET}")
    
    # Task 실행
    all_results = {}
    task_functions = {
        "vqav2": eval_vlm_vqav2,
        "gqa": eval_vlm_gqa,
        "textvqa": eval_vlm_textvqa,
        "okvqa": eval_vlm_ok_vqa,
    }
    
    for task in tasks:
        if task in task_functions:
            print(f"\n{Fore.MAGENTA}>>> Running {task.upper()} <<<{Fore.RESET}")
            
            try:
                results = task_functions[task](
                    model,
                    processor,
                    num_samples=num_samples,
                    batch_size=batch_size
                )
                all_results.update(results)
                
            except Exception as e:
                logging.error(f"{Fore.RED}Task {task} failed: {e}{Fore.RESET}")
                all_results[f"{task}_error"] = str(e)
        else:
            logging.warning(f"{Fore.YELLOW}Unknown task: {task}{Fore.RESET}")
    
    # 평균 계산
    accuracies = [v for k, v in all_results.items() if k.endswith('_accuracy')]
    if accuracies:
        all_results['average_accuracy'] = sum(accuracies) / len(accuracies)
        print(
            f"\n{Fore.GREEN}Average Accuracy: {all_results['average_accuracy']:.4f}{Fore.RESET}"
        )
    
    # 결과 저장
    os.makedirs('vlm_benchmark_results', exist_ok=True)
    result_path = f'vlm_benchmark_results/{os.path.basename(model_path)}.json'
    
    with open(result_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"{Fore.GREEN}Results saved to: {result_path}{Fore.RESET}")
    
    return all_results


def read_config(config_path: str) -> dict:
    """Read YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_from_config() -> None:
    """Run VLM evaluation from configuration file."""
    parser = argparse.ArgumentParser(
        description="Run VLM evaluation based on configuration file."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file.",
    )
    args = parser.parse_args()
    
    try:
        config = read_config(args.config)
        vlm_evaluator(**config)
    except Exception as e:
        print(f"{Fore.RED}Evaluation failed: {str(e)}{Fore.RESET}")
        raise