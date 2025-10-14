import argparse
import os
import json
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple, Iterable, Set
import random

import re
import numpy as np
from PIL import Image
from tqdm import tqdm, trange

import torch
from transformers import AutoModelForCausalLM
#from torch.utils.data import Dataset, DataLoader



#-------------------
#1) CLI (mirrors GenEval)
#------------------


def parse_args():
    p = argparse.ArgumentParser(description="GenEval generator for Ovis-U1-3B (txt->img)")
    p.add_argument("metadata_file", type=str, help="JSONL with one metadata dict per line (has 'prompt')")
    #p.add_argument("--model_path", type=str, default="AIDC-AI/Ovis-U1-3B", help="HF repo or local path for Ovis-U1-3B (trust_remote_code=True)")
    p.add_argument("--model_path", type=str, default="/data/nfs_home/ramanara/2025/MLLM/geneval/Ovis-U1/Ovis-U1-3B", help="HF repo or local path for Ovis-U1-3B (trust_remote_code=True)")
    p.add_argument("--outdir", type=str, default="outputs", help="Output root directory")

    # Resume + line-range control
    p.add_argument("--start-line", type=int, default=None, help="Start line number (inclusive). If omitted, inferred.")
    p.add_argument("--end-line", type=int, default=None, help="End line number (inclusive). If omitted, inferred.")
    p.add_argument("--resume", action="store_true", help="Resume: skip line indices that already have outputs in --outdir")
    p.add_argument("--line-base", type=int, choices=[0, 1], default=0, help="Interpret the first line as 0 (default) or 1.")

    # File naming / padding
    p.add_argument("--index-pad", type=int, default=5, help="Zero-pad width for index folders (e.g., 5 → 00037)")
    p.add_argument("--sample-pad", type=int, default=5, help="Zero-pad width for sample files (e.g., 5 → 00000.png)")

    # Generation Knobs
    p.add_argument("--n_samples", type=int, default=4, help="Images per prompt")
    p.add_argument("--steps", type=int, default=50, help="Diffusion steps (num_steps)")
    p.add_argument("--H", type=int, default=1024, help="Height")
    p.add_argument("--W", type=int, default=1024, help="Width")
    p.add_argument("--scale", type=float, default=5.0, help="Text guidance (txt_cfg)")
    p.add_argument("--seed", type=int, default=42, help="Global base seed")

    # Batch size in Ovis native path is effectively 1 (deterministic); keep flag for API symmetry
    p.add_argument("--batch_size", type=int, default=1, help="Unused (kept for parity)")

    # Optional: turn off grid saving like GenEval (we don’t create grids here)
    p.add_argument("--skip_grid", action="store_true", help="(No-op for Ovis) Kept for parity")

    return p.parse_args()



# -----------------------------------------
# 4) JSONL slicing & resume support
# -----------------------------------------

def jsonl_last_index(path:str, line_base: int)-> int:
    """Return the last line index in a JSONL (respecting line_base)"""
    last = line_base - 1
    with open(path, "r", encoding="utf-8") as f:
        for i, _ in enumerate(f, start=line_base):
            last = i
    return last

def scan_processed_indices(outdir: str,  n_samples: int) -> Set[int]:
    """
    Parse existing index folders in outdir and consider an index 'processed'
    only if samples/ contains >= n_samples PNGs. Returns a set of ints.
    """

    done: Set[int] = set()
    if not os.path.isdir(outdir):
        return done

    for name in os.listdir(outdir):
        path = os.path.join(outdir, name)
        if not os.path.isdir(path):
            continue
        
        #accept either zero-pad or plain numeric folder names
        if not re.fullmatch(r"\d+", name):
            continue

        idx = int(name) # handles "00037" -> 37 as well
        samples_dir = os.path.join(path, "samples")
        if os.path.isdir(samples_dir):
            pngs = [f for f in os.listdir(samples_dir) if f.lower().endswith(".png")]

            if len(pngs) >= n_samples:
                done.add(idx)

    return done

def stream_jsonl_slice(path: str, start_line:int, end_line: int, line_base: int) -> Iterable[tuple[int, Dict[str, Any]]]:
    """
        Stream only the requested inclusive line range from a large JSONL, without loading the entire file.
        Yields: (true_index, record_dict), where true_index is the JSONL line number respecting line_base.
    """
    assert end_line >= start_line, "end_line must be >= start_line"
    with open (path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=line_base):
            if idx < start_line:
                continue
            if idx > end_line:
                break
            rec = json.loads(line)
            yield idx, rec


# ----------------------------------------------------
# 2) Utilities copied/trimmed from your Ovis script
#    (unchanged logic; just wrapped into functions)
# ----------------------------------------------------


def _load_blank_image(width: int, height: int) -> Image.Image:
    """ Builds a white canvas used by ovis"""
    return Image.new("RGB", (width, height), (255, 255, 255)).convert("RGB")

def _build_inputs(
    model,
    text_tokenizer,
    visual_tokenizer,
    prompt: str,
    pil_image: Image.Image,
    target_width: int,
    target_height: int,
   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    if pil_image is not None:
        target_size = (int(target_width), int(target_height))
        pil_image, vae_pixel_values, cond_img_ids = model.visual_generator.process_image_aspectratio(
                pil_image, target_size
                )

        cond_img_ids[..., 0]= 1.0
        vae_pixel_values = vae_pixel_values.unsqueeze(0).to(device=model.device)
        width = pil_image.width
        height = pil_image.height
        resized_height, resized_width = visual_tokenizer.smart_resize(
                height, width, max_pixels = visual_tokenizer.image_processor.min_pixels
                )
        pil_image = pil_image.resize((resized_width, resized_height))

    else:
        vae_pixel_values  = None
        cond_img_ids = None


    prompt, input_ids, pixel_values, grid_thws = model.preprocess_inputs(
            prompt,
            [pil_image],
            generation_preface = None,
            return_labels = False,
            propagate_exception = False,
            multimodal_type = "single_image",
            fix_sample_overall_length_navit = False,
            )

    attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
    input_ids = input_ids.unsqueeze(0).to(device=model.device)
    attention_mask = attention_mask.unsqueeze(0).to(device=model.device)

    if pixel_values is not None:
        pixel_values = torch.cat(
                [pixel_values.to(device=visual_tokenizer.device, dtype=torch.bfloat16)], dim=0
        )

    if grid_thws is not None:
        grid_thws = torch.cat([grid_thws.to(device=visual_tokenizer.device)], dim=0)

    return input_ids, pixel_values, attention_mask, grid_thws, vae_pixel_values


def _pipe_t2i(
    model,
    prompt: str,
    height: int,
    width: int,
    steps: int,
    cfg: float,
    seed: int = 42,
) -> List[Image.Image]:
    """
    Native Ovis-U1 txt→img generation path (faithful to your test_txt_to_img.py):
    - Builds unconditional & conditional states
    - Calls model.generate_condition / model.generate_img
    - Returns list of PIL.Image (we return [img] here)
    """
    text_tokenizer = model.get_text_tokenizer()
    visual_tokenizer = model.get_visual_tokenizer()
    gen_kwargs = dict(
        max_new_tokens=1024,
        do_sample=False,
        top_p=None,
        top_k=None,
        temperature=None,
        repetition_penalty=None,
        eos_token_id=text_tokenizer.eos_token_id,
        pad_token_id=text_tokenizer.pad_token_id,
        use_cache=True,
        height=height,
        width=width,
        num_steps=steps,
        seed=seed,
        img_cfg=0,
        txt_cfg=cfg,
    )

    # Unconditional branch
    uncond_image = _load_blank_image(width, height)
    uncond_prompt = "<image>\nGenerate an image."
    input_ids, pixel_values, attention_mask, grid_thws, _ = _build_inputs(
        model, text_tokenizer, visual_tokenizer, uncond_prompt, uncond_image, width, height
    )
    with torch.inference_mode():
        no_both_cond = model.generate_condition(
            input_ids, pixel_values=pixel_values, attention_mask=attention_mask, grid_thws=grid_thws, **gen_kwargs
        )

    # Conditional branch (prepend instruction like your script)
    rich_prompt = (
        "<image>\nDescribe the image by detailing the color, shape, size, texture, quantity, text, "
        "and spatial relationships of the objects:" + prompt
    )
    no_txt_cond = None
    input_ids, pixel_values, attention_mask, grid_thws, vae_pixel_values = _build_inputs(
        model, text_tokenizer, visual_tokenizer, rich_prompt, uncond_image, width, height
    )
    with torch.inference_mode():
        cond = model.generate_condition(
            input_ids, pixel_values=pixel_values, attention_mask=attention_mask, grid_thws=grid_thws, **gen_kwargs
        )
        cond["vae_pixel_values"] = vae_pixel_values
        images = model.generate_img(cond=cond, no_both_cond=no_both_cond, no_txt_cond=no_txt_cond, **gen_kwargs)

    # model.generate_img returns a list/iterable of PIL images; keep a list for uniformity
    return images if isinstance(images, list) else [images]


# -----------------------------------------
# 3) Minimal seed control (determinism)
# -----------------------------------------
def set_global_seed(seed: int):
    random.seed(seed)
    np = __import__("numpy")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _round_to_mult8(x: int)-> int:
    return int(round(x / 8) * 8)


# -----------------------------------------
# 5) Main: GenEval loop → Ovis generation
# -----------------------------------------
def main():
    args = parse_args()

    # Infer start/end if omitted
    inferred_start = args.line_base if args.start_line is None else args.start_line
    inferred_end = jsonl_last_index(args.metadata_file, args.line_base) if args.end_line is None else args.end_line
    if inferred_end < inferred_start:
        raise ValueError("Final end-line must be >= start-line after inference.")

    os.makedirs(args.outdir, exist_ok=True)

    # Resume bookkeeping
    processed_indices = scan_processed_indices(args.outdir, args.n_samples) if args.resume else set()

    # If resuming with no explicit start, jump to first missing index
    if args.resume and args.start_line is None:
        cur = inferred_start
        while cur in processed_indices and cur <= inferred_end:
            cur += 1
        inferred_start = cur

    # Build Ovis-U1-3B model once
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model, _ = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        output_loading_info=True,
        trust_remote_code=True,
    )
    model = model.eval().to(device)
    model = model.to(dtype)

    # HW-friendly sizes
    H = _round_to_mult8(args.H)
    W = _round_to_mult8(args.W)

    seen_indices: Set[int] = set(processed_indices)  # guard against overlaps/duplication

    # Iterate ONLY the requested slice; folder names = TRUE JSONL line numbers
    for true_idx, metadata in tqdm(
        stream_jsonl_slice(args.metadata_file, inferred_start, inferred_end, args.line_base),
        desc=f"Lines {inferred_start}-{inferred_end} (base={args.line_base})"
    ):
        # Skip already-complete indices (resume mode)
        if args.resume and true_idx in processed_indices:
            continue

        if true_idx in seen_indices:
            raise RuntimeError(f"Duplicate line index encountered: {true_idx}")
        seen_indices.add(true_idx)

        # Determinism: base seed per-line; per-sample adds +k
        set_global_seed(args.seed)

        # Folder: ORIGINAL line number, zero-padded
        idx_str = str(true_idx).zfill(args.index_pad)
        outpath = os.path.join(args.outdir, idx_str)
        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)

        # Save the single-line metadata.jsonl (unaltered)
        with open(os.path.join(outpath, "metadata.jsonl"), "w", encoding="utf-8") as fp:
            fp.write(json.dumps(metadata) + "\n")

        prompt = metadata["prompt"]

        # Generate n_samples; filenames 0000.png.. with requested padding
        for k in range(args.n_samples):
            set_global_seed(args.seed + k)
            images = _pipe_t2i(
                model=model,
                prompt=prompt,
                height=H,
                width=W,
                steps=args.steps,
                cfg=args.scale,
                seed=args.seed + k,
            )
            img = images[0]
            fname = f"{str(k).zfill(args.sample_pad)}.png"
            img.save(os.path.join(sample_path, fname))

    print(f"[DONE] Wrote outputs for lines [{inferred_start}..{inferred_end}] (base={args.line_base}) to '{args.outdir}'.")


if __name__ == "__main__":
    main()




