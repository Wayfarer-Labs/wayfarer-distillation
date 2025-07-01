import torch
import re
from datasets import load_dataset
from typing import List, Dict, Any

def contains_cli_args(prompt: str) -> bool:
    """Check if prompt contains command line arguments like --ar 16:9, --style, etc."""
    cli_patterns = [
        r'--\w+',           # --ar, --style, etc.
        r'-\w+\s+\d+:\d+',  # -ar 16:9
        r'--\w+\s+\w+',     # --style cinematic
    ]
    return any(re.search(pattern, prompt, re.IGNORECASE) for pattern in cli_patterns)

def is_valid_prompt(prompt: str, toxicity: float, obscene: float, identity_attack: float, 
                   insult: float, threat: float, sexual_explicit: float, 
                   min_length: int = 20, max_nsfw: float = 0.01) -> bool:
    """Filter prompts based on Self-Forcing paper criteria + additional strict filtering.
    
    Filters out prompts with:
    - Less than 20 characters or 10 words
    - CLI arguments (--ar, --style, etc.)
    - NSFW content > 0.01 threshold
    - Unwanted characters: & # ; = < >
    - Unwanted words: quot, fps, 4k, 8k, -seed, prompt, -motion, -gs, 16:9, font:
    - "Message: 1 Attachment" phrases
    """
    # Basic validation - must be non-empty string
    if not prompt or not isinstance(prompt, str):
        return False
    
    # Strip whitespace for consistent checking
    prompt_stripped = prompt.strip()
    if not prompt_stripped:
        return False
    
    # Length check - use stripped version
    if len(prompt_stripped) < min_length:
        return False
    
    # Word count check - at least 10 words
    words = prompt_stripped.split()
    if len(words) < 10:
        return False
    
    # Debug: Double-check word count
    actual_word_count = len([w for w in words if w.strip()])
    if actual_word_count < 10:
        return False
    
    # CLI arguments check
    if contains_cli_args(prompt_stripped):
        return False
    
    # NSFW checks - any category above threshold fails
    nsfw_scores = [toxicity, obscene, identity_attack, insult, threat, sexual_explicit]
    if any(score > max_nsfw for score in nsfw_scores):
        return False
    
    # Additional filtering criteria
    prompt_lower = prompt_stripped.lower()
    
    # Check for unwanted phrases
    if "message: 1 attachment" in prompt_lower:
        return False
    
    # Check for unwanted characters
    if any(char in prompt_stripped for char in ['&', '#', ';', '=', '<', '>']):
        return False
    
    # Check for unwanted words
    unwanted_words = ['quot', 'fps', '4k', '8k', '-seed', 'prompt', '-motion', '-gs', '16:9', 'font:']
    if any(word in prompt_lower for word in unwanted_words):
        return False
    
    return True

def filter_prompts_batch(data: Dict[str, List[Any]]) -> List[str]:
    """Filter a batch of prompts and return only valid ones."""
    valid_prompts = []
    
    for i in range(len(data['prompt'])):
        prompt = data['prompt'][i]
        
        # Skip if prompt is None or not a string
        if not prompt or not isinstance(prompt, str):
            continue
            
        # Strip whitespace
        prompt_clean = prompt.strip()
        if not prompt_clean:
            continue
        
        if is_valid_prompt(
            prompt=prompt_clean,
            toxicity=data['toxicity'][i],
            obscene=data['obscene'][i], 
            identity_attack=data['identity_attack'][i],
            insult=data['insult'][i],
            threat=data['threat'][i],
            sexual_explicit=data['sexual_explicit'][i]
        ):
            valid_prompts.append(prompt_clean)  # Use cleaned version
    
    return valid_prompts

def create_text_prompts_vidprom(output_path: str, num_prompts: int = 16_000, 
                               seed: int = 42, batch_size: int = 50_000) -> None:
    """
    Create filtered text prompts from VidProM dataset.
    
    Args:
        output_path: Path to save filtered prompts
        num_prompts: Exact number of prompts needed
        seed: Random seed for reproducibility  
        batch_size: How many samples to fetch per iteration
    """
    print(f"Loading VidProM dataset...")
    ds = load_dataset("WenhaoWang/VidProM", split="train", cache_dir='./wayfarer_distillation/data/VidProM')
    dataset_size = len(ds)
    print(f"Dataset loaded with {dataset_size:,} total prompts")
    
    torch.manual_seed(seed)
    valid_prompts = []
    used_indices = set()
    
    print(f"Filtering to get exactly {num_prompts:,} valid prompts...")
    
    iteration = 0
    while len(valid_prompts) < num_prompts:
        iteration += 1
        needed = num_prompts - len(valid_prompts)
        
        # Sample more than needed since many will be filtered out
        # Estimate ~5-10% pass rate due to aggressive filtering
        sample_size = min(batch_size, max(needed * 12, 10_000))
        
        # Generate random indices, avoiding already used ones
        available_indices = [i for i in range(dataset_size) if i not in used_indices]
        if len(available_indices) < sample_size:
            print(f"Warning: Running out of unused data. Only {len(available_indices)} indices left.")
            sample_size = len(available_indices)
        
        if sample_size == 0:
            raise RuntimeError("Exhausted dataset without finding enough valid prompts!")
        
        # Sample random indices
        perm = torch.randperm(len(available_indices))[:sample_size]
        indices = [available_indices[i] for i in perm]
        used_indices.update(indices)
        
        # Fetch and filter batch
        print(f"Iteration {iteration}: Sampling {sample_size:,} prompts (need {needed:,} more)")
        data = ds[indices]
        batch_valid = filter_prompts_batch(data)
        
        valid_prompts.extend(batch_valid[:needed])  # Don't exceed target
        
        print(f"  Found {len(batch_valid):,} valid prompts in batch ({len(batch_valid)/sample_size:.1%} pass rate)")
        print(f"  Total valid prompts: {len(valid_prompts):,}/{num_prompts:,}")
        
        # Debug: Check word count of new batch
        if batch_valid:
            word_counts = [len(p.split()) for p in batch_valid]
            min_words = min(word_counts)
            print(f"  Batch validation: min words = {min_words}, max words = {max(word_counts)}")
            if min_words < 10:
                problematic = [p for p in batch_valid if len(p.split()) < 10]
                print(f"  ⚠️  {len(problematic)} prompts in batch have <10 words!")
    
    # Final validation before writing
    print(f"Performing final validation of {len(valid_prompts):,} prompts...")
    final_valid = []
    for prompt in valid_prompts:
        if prompt and prompt.strip() and len(prompt.split()) >= 10:
            final_valid.append(prompt)
        else:
            print(f"⚠️  Removing invalid prompt in final check: '{prompt}' ({len(prompt.split()) if prompt else 0} words)")
    
    if len(final_valid) != len(valid_prompts):
        print(f"⚠️  Final validation removed {len(valid_prompts) - len(final_valid)} invalid prompts")
        valid_prompts = final_valid
    
    # Write to file - ensure no empty lines
    print(f"Writing {len(valid_prompts):,} prompts to {output_path}")
    
    # Debug: Check final prompts before writing
    short_prompts = [p for p in valid_prompts if len(p.split()) < 10]
    if short_prompts:
        print(f"⚠️  WARNING: Found {len(short_prompts)} prompts with <10 words that passed filtering!")
        for p in short_prompts[:3]:  # Show first 3
            print(f"   Example: '{p}' ({len(p.split())} words)")
    
    with open(output_path, "w", encoding='utf-8') as f:
        for i, prompt in enumerate(valid_prompts):
            # Double-check prompt is valid before writing
            if prompt and prompt.strip():
                f.write(prompt.strip())
                # Add newline except for the last prompt
                if i < len(valid_prompts) - 1:
                    f.write("\n")
    
    print(f"✅ Successfully created {len(valid_prompts):,} filtered prompts!")
    
    # Print some stats
    total_sampled = len(used_indices)
    overall_pass_rate = len(valid_prompts) / total_sampled
    print(f"📊 Overall stats: {total_sampled:,} sampled → {len(valid_prompts):,} valid ({overall_pass_rate:.1%} pass rate)")
    print(f"🔍 Filters: ≥20 chars, ≥10 words, no CLI args, NSFW≤0.01, no chars [& # ; = < >], no words [quot fps 4k 8k -seed prompt -motion -gs 16:9 font:]")

if __name__ == "__main__":
    create_text_prompts_vidprom(
        output_path="filtered_text_prompts_16k.txt", 
        num_prompts=16_000, 
        seed=42
    )