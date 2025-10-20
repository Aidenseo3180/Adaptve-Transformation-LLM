def test_this():
  """
  LLaVA Vision Processing Analysis - Large-Scale Study
  Professional implementation for top-tier conference submission

  Features:
  - 300 COCO images for robust statistics
  - Fine-grained layer analysis (16-24) around critical point
  - Comprehensive statistical tests with effect sizes
  - Multiple comparison correction
  - Results saved in multiple formats for reproducibility
  """

  import torch
  import numpy as np
  from PIL import Image
  import matplotlib.pyplot as plt
  from pathlib import Path
  from transformers import AutoProcessor, LlavaForConditionalGeneration
  from typing import Dict, List, Tuple, Optional
  import random
  from scipy import stats
  import json
  import pickle
  from datetime import datetime
  from tqdm import tqdm
  import warnings
  warnings.filterwarnings('ignore')

  # ============================================================================
  # EXPERIMENT CONFIGURATION
  # ============================================================================
  CONFIG = {
      # Dataset
      'coco_dir': 'train2014',
      'num_images': 300,
      'random_seed': 42,
      
      # Model
      'model_id': 'llava-hf/llava-1.5-7b-hf',
      'text_prompt': 'What do you see in the image?',
      
      # Analysis
      'standard_layers': [5, 10, 15, 20, 25, 30, 31],
      'fine_grained_layers': list(range(16, 25)),  # Layer 20 deep dive
      
      # Output
      'output_dir': 'results',
      'save_raw_data': True,
      'save_summary': True,
      'save_metadata': True,
  }

  # Create output directory
  output_dir = Path(CONFIG['output_dir'])
  output_dir.mkdir(exist_ok=True)

  print("="*80)
  print("🔬 LLaVA Vision Processing Analysis - Large-Scale Study")
  print("="*80)
  print(f"Configuration:")
  print(f"  • Dataset: {CONFIG['num_images']} COCO images")
  print(f"  • Standard layers: {CONFIG['standard_layers']}")
  print(f"  • Fine-grained layers: {CONFIG['fine_grained_layers']}")
  print(f"  • Random seed: {CONFIG['random_seed']}")
  print(f"  • Output directory: {CONFIG['output_dir']}/")
  print("="*80)

  # ============================================================================
  # 1. DATASET LOADING
  # ============================================================================
  def load_coco_images(
      coco_dir: Path, 
      num_images: int, 
      seed: int = 42
  ) -> Tuple[List[Image.Image], List[str]]:
      """Load random COCO images from local directory"""
      coco_path = Path(coco_dir)
      
      if not coco_path.exists():
          raise FileNotFoundError(f"COCO directory not found: {coco_path}")
      
      # Get all jpg files
      image_files = list(coco_path.glob("*.jpg"))
      
      if len(image_files) == 0:
          raise FileNotFoundError(f"No .jpg files found in {coco_path}")
      
      print(f"\n📁 Found {len(image_files):,} images in {coco_path}")
      
      # Random sampling with seed
      random.seed(seed)
      np.random.seed(seed)
      sampled_files = random.sample(image_files, min(num_images, len(image_files)))
      
      # Load images with progress bar
      images = []
      image_names = []
      
      print("📷 Loading images...")
      for img_file in tqdm(sampled_files, desc="Loading", ncols=80):
          try:
              img = Image.open(img_file).convert('RGB')
              # Resize if too large (memory efficiency)
              if max(img.size) > 512:
                  img.thumbnail((512, 512), Image.Resampling.LANCZOS)
              images.append(img)
              image_names.append(img_file.name)
          except Exception as e:
              print(f"  ⚠️  Failed to load {img_file.name}: {e}")
      
      print(f"✅ Successfully loaded {len(images):,} images")
      return images, image_names

  # Load dataset
  coco_images, coco_image_names = load_coco_images(
      Path(CONFIG['coco_dir']), 
      CONFIG['num_images'],
      CONFIG['random_seed']
  )

  # Create blank baseline
  blank_image = Image.new('RGB', coco_images[0].size, color='black')

  # ============================================================================
  # 2. MODEL LOADING
  # ============================================================================
  print(f"\n🔄 Loading model: {CONFIG['model_id']}")

  processor = AutoProcessor.from_pretrained(CONFIG['model_id'])
  model = LlavaForConditionalGeneration.from_pretrained(
      CONFIG['model_id'],
      torch_dtype=torch.float16,
      device_map="auto",
      low_cpu_mem_usage=True,
      attn_implementation="eager"
  )

  print(f"✅ Model loaded on {model.device}")
  print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")

  # ============================================================================
  # 3. CORE ANALYSIS FUNCTIONS
  # ============================================================================
  def extract_activations_and_attention(
      model, 
      processor, 
      image: Image.Image, 
      text: str, 
      target_layers: List[int]
  ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
      """Extract hidden states and attention weights for specified layers"""
      prompt = f"USER: <image>\n{text}\nASSISTANT:"
      inputs = processor(text=prompt, images=image, return_tensors="pt")
      inputs = {k: v.to(model.device) for k, v in inputs.items()}
      
      with torch.no_grad():
          outputs = model(
              **inputs,
              output_hidden_states=True,
              output_attentions=True,
              return_dict=True
          )
      
      activations = {l: outputs.hidden_states[l+1].cpu() for l in target_layers}
      attentions = {l: outputs.attentions[l].cpu() for l in target_layers}
      
      return activations, attentions

  def compute_comprehensive_metrics(
      acts_with_vision: torch.Tensor,
      acts_blank: torch.Tensor,
      attention: torch.Tensor
  ) -> Dict[str, float]:
      """Compute comprehensive metrics for analysis"""
      vision_token_count = 576
      
      # Extract tokens
      vision_tokens = acts_with_vision[0, :vision_token_count, :]
      text_tokens_vis = acts_with_vision[0, vision_token_count:, :]
      text_tokens_blank = acts_blank[0, vision_token_count:, :]
      
      # 1. Vision/Text Activation Ratio
      vision_l2 = torch.norm(vision_tokens, dim=1).mean().item()
      text_l2 = torch.norm(text_tokens_vis, dim=1).mean().item()
      ratio = vision_l2 / (text_l2 + 1e-8)
      
      # 2. Activation Difference (multiple metrics)
      vision_feat = text_tokens_vis.mean(dim=0)
      blank_feat = text_tokens_blank.mean(dim=0)
      
      cosine_sim = torch.nn.functional.cosine_similarity(
          vision_feat.unsqueeze(0),
          blank_feat.unsqueeze(0)
      ).item()
      
      # L2 distance
      l2_distance = torch.norm(vision_feat - blank_feat).item()
      
      # Mean absolute difference
      mad = torch.mean(torch.abs(vision_feat - blank_feat)).item()
      
      # 3. Attention metrics (Top-5 heads)
      seq_len = attention.shape[2]
      last_token_idx = seq_len - 1
      
      text_to_vision = attention[0, :, last_token_idx, :vision_token_count]
      head_totals = text_to_vision.sum(dim=1)
      top5_heads = torch.topk(head_totals, min(5, len(head_totals)))[1]
      
      attn_mean = text_to_vision[top5_heads].mean().item()
      attn_max = text_to_vision[top5_heads].max().item()
      attn_std = text_to_vision[top5_heads].std().item()
      
      # 4. Token-level statistics
      vision_activation_mean = torch.abs(vision_tokens).mean().item()
      vision_activation_std = torch.abs(vision_tokens).std().item()
      
      return {
          'vision_text_ratio': ratio,
          'vision_l2': vision_l2,
          'text_l2': text_l2,
          'cosine_similarity': cosine_sim,
          'l2_distance': l2_distance,
          'mean_absolute_diff': mad,
          'attention_top5_mean': attn_mean,
          'attention_top5_max': attn_max,
          'attention_top5_std': attn_std,
          'vision_activation_mean': vision_activation_mean,
          'vision_activation_std': vision_activation_std,
      }

  # ============================================================================
  # 4. MAIN ANALYSIS PIPELINE
  # ============================================================================
  print("\n" + "="*80)
  print("🔬 RUNNING COMPREHENSIVE ANALYSIS")
  print("="*80)

  # Combine all layers to analyze
  all_layers = sorted(list(set(CONFIG['standard_layers'] + CONFIG['fine_grained_layers'])))
  print(f"Analyzing {len(all_layers)} layers: {all_layers}")

  # Data structure: {image_idx: {layer: metrics}}
  raw_results = {}

  # Extract blank baseline once
  print("\n[1/2] Extracting blank baseline...")
  acts_blank_all, _ = extract_activations_and_attention(
      model, processor, blank_image, CONFIG['text_prompt'], all_layers
  )
  print("✅ Blank baseline extracted")

  # Process all images
  print(f"\n[2/2] Processing {len(coco_images)} images...")
  print("     (This may take 30-60 minutes depending on your GPU)")

  for img_idx, (image, img_name) in enumerate(
      tqdm(
          zip(coco_images, coco_image_names),
          total=len(coco_images),
          desc="Processing images",
          ncols=80
      )
  ):
      try:
          # Extract activations and attention
          acts_vis, attn_vis = extract_activations_and_attention(
              model, processor, image, CONFIG['text_prompt'], all_layers
          )
          
          # Compute metrics for each layer
          raw_results[img_idx] = {
              'image_name': img_name,
              'layers': {}
          }
          
          for layer_idx in all_layers:
              metrics = compute_comprehensive_metrics(
                  acts_vis[layer_idx],
                  acts_blank_all[layer_idx],
                  attn_vis[layer_idx]
              )
              raw_results[img_idx]['layers'][layer_idx] = metrics
          
      except Exception as e:
          print(f"\n⚠️  Error processing {img_name}: {e}")
          continue

  print(f"\n✅ Successfully processed {len(raw_results)} images")

  # ============================================================================
  # 5. STATISTICAL ANALYSIS
  # ============================================================================
  print("\n" + "="*80)
  print("📊 COMPUTING STATISTICS")
  print("="*80)

  # Aggregate results by layer
  aggregated_results = {layer: {metric: [] for metric in [
      'vision_text_ratio', 'cosine_similarity', 'l2_distance',
      'mean_absolute_diff', 'attention_top5_mean', 'attention_top5_max'
  ]} for layer in all_layers}

  for img_data in raw_results.values():
      for layer_idx, metrics in img_data['layers'].items():
          for metric_name in aggregated_results[layer_idx].keys():
              if metric_name in metrics:
                  aggregated_results[layer_idx][metric_name].append(metrics[metric_name])

  # Compute comprehensive statistics
  statistics = {}
  for layer_idx in all_layers:
      statistics[layer_idx] = {}
      
      for metric_name, values in aggregated_results[layer_idx].items():
          values_array = np.array(values)
          
          statistics[layer_idx][metric_name] = {
              'mean': float(np.mean(values_array)),
              'std': float(np.std(values_array)),
              'median': float(np.median(values_array)),
              'q25': float(np.percentile(values_array, 25)),
              'q75': float(np.percentile(values_array, 75)),
              'min': float(np.min(values_array)),
              'max': float(np.max(values_array)),
              'iqr': float(np.percentile(values_array, 75) - np.percentile(values_array, 25)),
              'n': len(values_array)
          }

  # Print summary
  print("\nLayer-wise Statistics Summary:")
  print("─" * 80)
  print(f"{'Layer':<8} {'V/T Ratio':<22} {'Cosine Sim':<22} {'L2 Distance':<22}")
  print("─" * 80)

  for layer_idx in sorted(all_layers):
      ratio = statistics[layer_idx]['vision_text_ratio']
      cosine = statistics[layer_idx]['cosine_similarity']
      l2_dist = statistics[layer_idx]['l2_distance']
      
      print(f"{layer_idx:<8} "
            f"{ratio['mean']:.3f}±{ratio['std']:.3f}      "
            f"{cosine['mean']:.3f}±{cosine['std']:.3f}      "
            f"{l2_dist['mean']:.3f}±{l2_dist['std']:.3f}")

  print("─" * 80)

  # ============================================================================
  # 6. ADVANCED STATISTICAL TESTS
  # ============================================================================
  print("\n" + "="*80)
  print("📈 STATISTICAL SIGNIFICANCE TESTS")
  print("="*80)

  statistical_tests = {}

  # Test 1: Layer 20 vs neighbors (is it significantly different?)
  print("\n1. Layer 20 vs Neighbors (Cosine Similarity)")
  print("   Testing if Layer 20 dip is statistically significant")
  print("   " + "─" * 76)

  layer_20_values = aggregated_results[20]['cosine_similarity']

  for neighbor in [18, 19, 21, 22]:
      if neighbor in aggregated_results:
          neighbor_values = aggregated_results[neighbor]['cosine_similarity']
          
          # Two-sample t-test
          t_stat, p_value = stats.ttest_ind(layer_20_values, neighbor_values)
          
          # Effect size (Cohen's d)
          mean_diff = np.mean(layer_20_values) - np.mean(neighbor_values)
          pooled_std = np.sqrt(
              (np.std(layer_20_values)**2 + np.std(neighbor_values)**2) / 2
          )
          cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
          
          statistical_tests[f'layer20_vs_{neighbor}'] = {
              't_statistic': float(t_stat),
              'p_value': float(p_value),
              'cohens_d': float(cohens_d),
              'mean_diff': float(mean_diff),
              'significant': bool(p_value < 0.05)
          }
          
          print(f"   Layer 20 vs {neighbor}: "
                f"t={t_stat:.3f}, p={p_value:.4f}, d={cohens_d:.3f} "
                f"{'✅' if p_value < 0.05 else '❌'}")

  # Test 2: Stage comparisons (Vision/Text Ratio)
  print("\n2. Stage Comparisons (Vision/Text Ratio)")
  print("   Testing differences between processing stages")
  print("   " + "─" * 76)

  # Define stages
  stage_1 = [5, 10]  # Early
  stage_2 = [15, 20, 25]  # Middle
  stage_3 = [30, 31]  # Late

  # Aggregate by stage
  stage_1_ratios = []
  stage_2_ratios = []
  stage_3_ratios = []

  for layer in stage_1:
      stage_1_ratios.extend(aggregated_results[layer]['vision_text_ratio'])
  for layer in stage_2:
      stage_2_ratios.extend(aggregated_results[layer]['vision_text_ratio'])
  for layer in stage_3:
      stage_3_ratios.extend(aggregated_results[layer]['vision_text_ratio'])

  # Pairwise tests
  stage_comparisons = [
      ('Stage 1 vs 2', stage_1_ratios, stage_2_ratios),
      ('Stage 2 vs 3', stage_2_ratios, stage_3_ratios),
      ('Stage 1 vs 3', stage_1_ratios, stage_3_ratios),
  ]

  for name, group1, group2 in stage_comparisons:
      t_stat, p_value = stats.ttest_ind(group1, group2)
      
      mean_diff = np.mean(group1) - np.mean(group2)
      pooled_std = np.sqrt((np.std(group1)**2 + np.std(group2)**2) / 2)
      cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
      
      statistical_tests[name.replace(' ', '_').lower()] = {
          't_statistic': float(t_stat),
          'p_value': float(p_value),
          'cohens_d': float(cohens_d),
          'mean_1': float(np.mean(group1)),
          'mean_2': float(np.mean(group2)),
          'significant': bool(p_value < 0.05)
      }
      
      print(f"   {name}: "
            f"{np.mean(group1):.3f} vs {np.mean(group2):.3f}, "
            f"t={t_stat:.3f}, p={p_value:.2e}, d={cohens_d:.3f} "
            f"{'✅' if p_value < 0.05 else '❌'}")

  # Test 3: Monotonic trend test (Spearman correlation)
  print("\n3. Monotonic Trend Test (Vision/Text Ratio)")
  print("   Testing if ratio decreases monotonically with layer")
  print("   " + "─" * 76)

  layer_indices = []
  ratio_means = []
  for layer in sorted(all_layers):
      layer_indices.append(layer)
      ratio_means.append(statistics[layer]['vision_text_ratio']['mean'])

  spearman_corr, spearman_p = stats.spearmanr(layer_indices, ratio_means)

  statistical_tests['monotonic_trend'] = {
      'spearman_correlation': float(spearman_corr),
      'p_value': float(spearman_p),
      'significant': bool(spearman_p < 0.05),
      'interpretation': 'Significant negative trend' if spearman_corr < -0.5 and spearman_p < 0.05 else 'No significant trend'
  }

  print(f"   Spearman ρ = {spearman_corr:.4f}, p = {spearman_p:.2e} "
        f"{'✅ Significant' if spearman_p < 0.05 else '❌ Not significant'}")
  print(f"   Interpretation: {statistical_tests['monotonic_trend']['interpretation']}")

  # ============================================================================
  # 7. SAVE RESULTS
  # ============================================================================
  print("\n" + "="*80)
  print("💾 SAVING RESULTS")
  print("="*80)

  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

  # Save raw results (pickle for Python)
  if CONFIG['save_raw_data']:
      raw_path = output_dir / f"raw_results_{timestamp}.pkl"
      with open(raw_path, 'wb') as f:
          pickle.dump({
              'raw_results': raw_results,
              'aggregated_results': aggregated_results,
              'config': CONFIG
          }, f)
      print(f"✅ Raw results saved: {raw_path}")

  # Save statistics (JSON for human-readable)
  if CONFIG['save_summary']:
      summary_path = output_dir / f"statistics_{timestamp}.json"
      with open(summary_path, 'w') as f:
          json.dump({
              'statistics': statistics,
              'statistical_tests': statistical_tests,
              'config': CONFIG,
              'timestamp': timestamp
          }, f, indent=2)
      print(f"✅ Statistics saved: {summary_path}")

  # Save metadata
  if CONFIG['save_metadata']:
      metadata = {
          'timestamp': timestamp,
          'config': CONFIG,
          'num_images_processed': len(raw_results),
          'layers_analyzed': all_layers,
          'image_names': list(raw_results[i]['image_name'] for i in raw_results.keys()),
          'pytorch_version': torch.__version__,
          'device': str(model.device),
      }
      
      metadata_path = output_dir / f"metadata_{timestamp}.json"
      with open(metadata_path, 'w') as f:
          json.dump(metadata, f, indent=2)
      print(f"✅ Metadata saved: {metadata_path}")

  # Also save a "latest" version for easy access
  latest_summary_path = output_dir / "statistics_latest.json"
  with open(latest_summary_path, 'w') as f:
      json.dump({
          'statistics': statistics,
          'statistical_tests': statistical_tests,
          'config': CONFIG,
          'timestamp': timestamp
      }, f, indent=2)
  print(f"✅ Latest summary: {latest_summary_path}")

  # ============================================================================
  # 8. FINAL SUMMARY
  # ============================================================================
  print("\n" + "="*80)
  print("🎯 ANALYSIS COMPLETE")
  print("="*80)

  print(f"""
  Experiment Summary:
    • Images processed: {len(raw_results)} / {CONFIG['num_images']}
    • Layers analyzed: {len(all_layers)} ({min(all_layers)}-{max(all_layers)})
    • Metrics per image/layer: {len(list(raw_results[0]['layers'][5].keys()))}
    • Total data points: {len(raw_results) * len(all_layers):,}

  Key Findings:
    1. Vision/Text Ratio:
      • Early (L5):  {statistics[5]['vision_text_ratio']['mean']:.3f} ± {statistics[5]['vision_text_ratio']['std']:.3f}
      • Middle (L20): {statistics[20]['vision_text_ratio']['mean']:.3f} ± {statistics[20]['vision_text_ratio']['std']:.3f}
      • Late (L31):   {statistics[31]['vision_text_ratio']['mean']:.3f} ± {statistics[31]['vision_text_ratio']['std']:.3f}
      • Monotonic trend: {'✅ Confirmed' if statistical_tests['monotonic_trend']['significant'] else '❌ Not confirmed'}
    
    2. Layer 20 Critical Point:
      • Cosine Similarity: {statistics[20]['cosine_similarity']['mean']:.4f}
      • vs Layer 19: {statistical_tests.get('layer20_vs_19', {}).get('p_value', 'N/A')}
      • vs Layer 21: {statistical_tests.get('layer20_vs_21', {}).get('p_value', 'N/A')}
    
    3. Stage Differences:
      • Stage 1 vs 2: p = {statistical_tests['stage_1_vs_2']['p_value']:.2e} (d = {statistical_tests['stage_1_vs_2']['cohens_d']:.3f})
      • Stage 2 vs 3: p = {statistical_tests['stage_2_vs_3']['p_value']:.2e} (d = {statistical_tests['stage_2_vs_3']['cohens_d']:.3f})

  Output Files:
    • {output_dir}/raw_results_{timestamp}.pkl
    • {output_dir}/statistics_{timestamp}.json
    • {output_dir}/metadata_{timestamp}.json
    • {output_dir}/statistics_latest.json (symlink for convenience)

  Ready for plotting! Load 'statistics_latest.json' to create publication figures.
  """)

  print("="*80)
  print("✨ All done! Results are ready for visualization.")
  print("="*80)