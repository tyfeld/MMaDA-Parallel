# MMaDA-Parallel
<h3>Parallel Multimodal Large Diffusion Language Models for Thinking-Aware Editing and Generation</h3></div>

<p align="center">
  <a href="https://arxiv.org/abs/2505.15809">
    <img
      src="https://img.shields.io/badge/MMaDA-Paper-red?logo=arxiv&logoColor=red"
      alt="MMaDA Paper on arXiv"
    />
  </a>
  <a href="https://huggingface.co/spaces/Gen-Verse/MMaDA">
    <img 
        src="https://img.shields.io/badge/MMaDA%20Demo-Hugging%20Face%20Space-blue?logo=huggingface&logoColor=blue" 
        alt="MMaDA on Hugging Face"
    />
  </a>
  <a href="https://huggingface.co/Gen-Verse/MMaDA-Parallel-Preview">
    <img 
        src="https://img.shields.io/badge/MMaDA--8B--Base-Hugging%20Face%20Model-orange?logo=huggingface&logoColor=yellow" 
        alt="MMaDA on Hugging Face"
    />
  </a>
</p>


## 🌌 Introduction
While thinking-aware generation aims to improve performance on complex tasks, we identify a critical failure mode where existing sequential, autoregressive approaches can paradoxically degrade performance due to error propagation. 
To systematically analyze this issue, we propose **ParaBench**, a new benchmark designed to evaluate both text and image output modalities. Our analysis using ParaBench reveals that this performance degradation is strongly correlated with poor alignment between the generated reasoning and the final image.
To resolve this, we propose a parallel multimodal diffusion framework that enables continuous, bidirectional interaction between text and images throughout the entire denoising trajectory. This model, **MMaDA-Parallel**, is trained with supervised finetuning and then further optimized by Parallel Reinforcement Learning (**ParaRL**), a novel strategy that applies semantic rewards along the trajectory to enforce cross-modal consistency. Experiments validate that our approach significantly improves cross-modal alignment and semantic consistency, achieving a 6.9\% improvement in **Output Alignment** on ParaBench compared to the state-of-the-art model, Bagel, establishing a more robust paradigm for thinking-aware image synthesis.


<!--



## Decoding Demo
We demonstrate the decoding process of MMaDA with a teaser video to show how a diffusion model generates text and image. The "Text Generation" part adopts a "semi-autoregressive" sampling method and the "MultiModal Generation" part adopts a non-autoregressive sampling method which is purely diffusion denoising.

<!-- <div style="display: flex; justify-content: center; flex-wrap: wrap;">
    <img src="assets/showcase0.8.gif" style="width: 90%" />
</div> -->

## 📰 Latest Updates 
* **[2025-11-02]** We release our [research paper](https://arxiv.org/abs/2505.15809) and [demo](https://huggingface.co/spaces/Gen-Verse/MMaDA) for Parallel Multimodal Large Diffusion Language Models for Thinking-Aware Editing and Generation.

## ⚙️ Quick Start
First, set up the enviroment:
```
pip install -r requirements.txt
```
Launch local Gradio demo:
```
python app.py
```
Or try it online via our [Huggingface Demo](https://huggingface.co/spaces/Gen-Verse/MMaDA-Parallel-Preview).

## 🚀 Inference
```bash
python generate.py
```

## 🔧 Training
training code will be released soon.


## 📊 Evaluation

Please refer to [evaluation/eval.md](evaluation/eval.md) for more details.

## 📖 Citation
```
@article{yang2025mmada,
  title={MMaDA: Multimodal Large Diffusion Language Models},
  author={Yang, Ling and Tian, Ye and Li, Bowen and Zhang, Xinchen and Shen, Ke and Tong, Yunhai and Wang, Mengdi},
  journal={arXiv preprint arXiv:2505.15809},
  year={2025}
}
```

## 🤝 Acknowledgments
This work is heavily based on [MMaDA](https://github.com/Gen-Verse/MMaDA) and [Lumina-DiMOO](https://github.com/Alpha-VLLM/Lumina-DiMOO). Thanks to all the authors for their great work.
