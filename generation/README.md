# Ovis-U1: Unified Understanding, Generation, and Editing Installation for GenEval Benchmarking

### Environment Set Up for Image Generation By Ovis-U1 (Works for both H100 & A100)

Set the environment for Ovis-U1 and GenEval benchamark using the below yaml file for GPUs (working for H100 & A100)

```bash
environment_geneval_ovis_h100.yaml
```

Clone the Ovis-U1 repo from here https://github.com/AIDC-AI/Ovis-U1.git inside the OvisU1-GenEval-Benchmark


Go inside the Ovis-U1 and download the snapshot of the Ovis-U1-3B using the following commands

```bash

cd Ovis-U1

# Check login (new CLI)
hf auth whoami

# Download the whole repo, pinned to main, into ./Ovis-U1-3B
hf download AIDC-AI/Ovis-U1-3B --revision main --local-dir ./Ovis-U1-3B

```

### Image Generation

Then use these commands to run the Ovis-U1 on the GenEval Dataset

```bash
python generation/ovis_u1_generation_complete.py   prompts/evaluation_metadata.jsonl   --outdir <IMAGE_FOLDER>/geneval_images_ovis_u1   --model_path Ovis-U1/Ovis-U1-3B   --n_samples 4 --steps 50 --H 1024 --W 1024 --scale 5.0 --seed 42   --resume  --line-base 0
```

Once the images are generated check whether they are in this format or not

to generate 4 images per prompt using Ovis-U1 and save in `<IMAGE_FOLDER>`.

The generated format should be
```
<IMAGE_FOLDER>/
    00000/
        metadata.jsonl
        samples/
            0000.png
            0001.png
            0002.png
            0003.png
    00001/
        ...
```
where `metadata.jsonl` contains the `N`-th line from `evaluation_metadata.jsonl`.


### Evaluation of Generated Images in A100 GPU Set Up 

Set up a separate environment for evaluation 

Follow this link for setting up the environment https://github.com/djghosh13/geneval/issues/12#issuecomment-2789202971

## or use the following methods or use the below environment file 

```bash
environment_geneval_evaluation_a100.yaml
```

The guide works perfectly for Ampere GPUs (like A6000, A100) by using the basic steps. For Hopper GPUs (like H100) there are certain additional things that you need to do which is stated at the necessary places.

First and foremost, the python version. It is quite important. I used python 3.8.10 and it worked like charm. So I would advise you on using that.

Given that you have installed python 3.8.10 and its alias is python I like to use venv for my environment creation. You can obviously use conda and do the same exact thing. Activate your environment and do the following things step by step.

# 1. Cloning the repository and downloading the ckpt

```
git clone https://github.com/djghosh13/geneval.git
cd geneval --> ./evaluation/download_models.sh "<OBJECT_DETECTOR_FOLDER>/" (this downloads the ckpt to <OBJECT_DETECTOR_FOLDER>/)
```

# 2. Installing dependencies

```python

# PyTorch + CUDA 12.1 (exact versions tested)
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu121

# If you hit networkx-related errors
pip install networkx==2.8.8

# CLIP + benchmarking
pip install open-clip-torch==2.26.1
pip install clip-benchmark

# Core utilities
pip install -U openmim
pip install einops
python -m pip install lightning
pip install diffusers transformers
pip install tomli
pip install platformdirs

# Make sure setuptools is up-to-date
pip install --upgrade setuptools

```

# 3. mmengine and mmcv dependency installation (Should be done after pip installations)

```bash
mim install mmengine mmcv-full==1.7.2
```

If you are using the newer Hopper GPUs please change this Step 3 by doing the following. (thanks to @rinongal's comment: A note on installing and running geneval #12 (comment))

```bash
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv; git checkout 1.x
MMCV_WITH_OPS=1 MMCV_CUDA_ARGS="-arch=sm_90" pip install -v -e .
```

# 4. mmdet installation

```bash
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection; git checkout 2.x
pip install -v -e . (In case of Hopper GPUs change this to:MMCV_CUDA_ARGS="-arch=sm_90" pip install -v -e .
```


### Evaluation 

For Evaluation on A100 use this following command

```bash
python evaluation/evaluate_images.py <IMAGE_FOLDER>/geneval_images_ovis_u1 --outfile Results/results.jsonl --model-path object_detection_folder

python evaluation/summary_scores.py Results/results.jsonl
```












