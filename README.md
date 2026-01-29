
<div align="center">
<h1>[ICLR 2026] MMaDA-Parallel: Parallel Multimodal Large Diffusion Language Models for Thinking-Aware Editing and Generation</h1>
<p align="center">
  <a href="https://arxiv.org/abs/2511.09611">
    <img
      src="https://img.shields.io/badge/MMaDA--Parallel-Paper-red?logo=arxiv&logoColor=red"
      alt="MMaDA-Parallel Paper on arXiv"
    />
  </a>
  <a href="https://tyfeld.github.io/mmadaparellel.github.io/">
    <img 
        src="https://img.shields.io/badge/Project_Website-Page-brightgreen?logo=safari" 
        alt="MMaDA Parallel Page"
    />
  </a></p><p>
  <a href="https://huggingface.co/tyfeld/MMaDA-Parallel-A">
    <img 
        src="https://img.shields.io/badge/MMaDA--Parallel--A-Hugging%20Face%20Model-orange?logo=huggingface&logoColor=yellow" 
        alt="MMaDA on Hugging Face"
    />
  </a>
    <a href="https://huggingface.co/tyfeld/MMaDA-Parallel-M">
    <img 
        src="https://img.shields.io/badge/MMaDA--Parallel--M-Hugging%20Face%20Model-orange?logo=huggingface&logoColor=yellow" 
        alt="MMaDA on Hugging Face"
    />
  </a>
  </a>
    <a href="https://huggingface.co/datasets/tyfeld/ParaBench">
    <img 
        src="https://img.shields.io/badge/ParaBench-BenchMark-orange?logo=huggingface&logoColor=yellow" 
        alt="MMaDA on Hugging Face"
    />
  </a>
</p>
</div>



## 🌌 Introduction

<div align="center">
    <img src="assets/demo.gif" alt="Parallel Generation Demo" style="width: 100%" />
    <p align="center">Demo: Parallel text-image generation in action.</p>
</div>


While thinking-aware generation aims to improve performance on complex tasks, we identify a critical failure mode where existing sequential, autoregressive approaches can paradoxically degrade performance due to error propagation. 
To systematically analyze this issue, we propose **ParaBench**, a new benchmark designed to evaluate both text and image output modalities. Our analysis using ParaBench reveals that this performance degradation is strongly correlated with poor alignment between the generated reasoning and the final image.
To resolve this, we propose a parallel multimodal diffusion framework that enables continuous, bidirectional interaction between text and images throughout the entire denoising trajectory. This model, **MMaDA-Parallel**, is trained with supervised finetuning and then further optimized by Parallel Reinforcement Learning (**ParaRL**), a novel strategy that applies semantic rewards along the trajectory to enforce cross-modal consistency. Experiments validate that our approach significantly improves cross-modal alignment and semantic consistency, achieving a 6.9\% improvement in **Output Alignment** on ParaBench compared to the state-of-the-art model, Bagel, establishing a more robust paradigm for thinking-aware image synthesis.
<div align="center" style="display: flex; justify-content: center; flex-wrap: wrap;">
    <img src="assets/method.png" style="width: 90%" />
    <p align="center">Architecture of MMaDA-Parallel. During Training, image and text responses are masked and predicted in parallel with a uniform mask predictor. During Sampling, the model performs parallel decoding to generate both image and text responses jointly, enabling continuous cross-modal interaction. </p>
</div>

## Results
<div align="center" style="display: flex; justify-content: center; flex-wrap: wrap;">
    <img src="assets/lumina_01.png" alt="Main Results" style="width: 90%" />
    <p align="center">Qualitative comparison. </p>
</div>


<div align="center" style="display: flex; justify-content: center; flex-wrap: wrap;">
    <img src="assets/mainresults.png" alt="Main Results" style="width: 90%" />
    <p align="center">Quantitative Results on ParaBench.</p>
</div>






## 📰 Latest Updates 
* **[2025-01-27]** [MMaDA-Parallel](https://arxiv.org/abs/2511.09611) is accepted by ICLR 2026.
* **[2025-11-11]** We release our codes and models for [MMaDA-Parallel](https://arxiv.org/abs/2511.09611), with two released 8B models [MMaDA-Parallel-A](https://huggingface.co/tyfeld/MMaDA-Parallel-A) and [MMaDA-Parallel-M](https://huggingface.co/tyfeld/MMaDA-Parallel-M).
* **[2025-11-10]** We release our [research paper](https://arxiv.org/abs/2511.09611) for Parallel Multimodal Large Diffusion Language Models for Thinking-Aware Editing and Generation.

## ⚙️ Quick Start
**Note:** Our model has been successfully validated on synthetic datasets focusing on **environments, still life, architecture, and natural landscapes.** Its performance on out-of-distribution inputs—such as human faces or real-world photographic imagery—has not yet been fully explored. We are actively expanding our training corpus to include more diverse datasets.


### 1. Environment Setup
First, start with a torch environment with torch 2.3.1 or higher version, then install the following dependencies:
```
pip install -r requirements.txt
```

We provide two varients of MMaDA-Parallel with different tokenizers. MMaDA-Parallel-A is trained with tokenizer Amused-VQ, and MMaDA-Parallel-M is trained with tokenizer Magvitv2.

### 2. Experiencing Parallel Gen with MMaDA-Parallel-A
You can directly use the local gradio app to experience the parallel generation with MMaDA-Parallel-A:
```bash
python app.py
```

Or you can use the inference script to generate the parallel generation results:

```bash
cd MMaDA-Parallel-A
python inference.py \
    --checkpoint tyfeld/MMaDA-Parallel-A \
    --vae_ckpt tyfeld/MMaDA-Parallel-A \
    --prompt "Replace the laptops with futuristic transparent tablets displaying holographic screens, and change the drink to a cup of glowing blue energy drink." \
    --image_path examples/image.png \
    --height 512 \
    --width 512 \
    --timesteps 64 \
    --text_steps 128 \
    --text_gen_length 256 \
    --text_block_length 32 \
    --cfg_scale 0 \
    --cfg_img 4.0 \
    --temperature 1.0 \
    --text_temperature 0 \
    --seed 42 \
    --output_dir output/results_interleave
```

### 3. Parallel Gen with MMaDA-Parallel-M
```bash
cd MMaDA-Parallel-M
python inference.py interleave_root=./interleave_validation  
```

## 🔧 Training
We provide our training code for MMaDA-Parallel-A here. 
### Data preparation
To train the model, we need to preprocess the dataset first.

Step 1: Replace the data path and data type in the scripts/run_pre_token.sh

Step 2: Run the Pre-Tokenizer

```
bash scripts/run_pre_token.sh
```
### Start training

Step 1: Update the training data path in configs/data.yaml with the  `all_records.json` obtained in data preparation.

Step 2: Replace the path in scripts/train.sh with the model checkpoint path.

Then start training with
```
bash scripts/train.sh
```

<!-- ## 📊 Evaluation
Please refer to [evaluation/eval.md](evaluation/eval.md) for more details. -->

## 📖 Citation
```
@article{tian2025mmadaparallel,
  title={MMaDA-Parallel: Multimodal Large Diffusion Language Models for Thinking-Aware Editing and Generation},
  author={Tian, Ye and Yang, Ling and Yang, Jiongfan and Wang, Anran and Tian, Yu and Zheng, Jiani and Wang, Haochen and Teng, Zhiyang and Wang, Zhuochen and Wang, Yinjie and Tong, Yunhai and Wang, Mengdi and Li, Xiangtai},
  journal={arXiv preprint arXiv:2511.09611},
  year={2025}
}
```

## 🤝 Acknowledgments
This work is heavily based on [MMaDA](https://github.com/Gen-Verse/MMaDA) and [Lumina-DiMOO](https://github.com/Alpha-VLLM/Lumina-DiMOO). Thanks to all the authors for their great work.
