# Improving the Throughput of Diffusion-based Large Language Models via a Training-Free Confidence-Aware Calibration

This repository provides the **official PyTorch implementation** of our paper:
> **"Improving the Throughput of Diffusion-based Large Language Models via a Training-Free Confidence-Aware Calibration"**  
> *Jucheng Shen, Gaurav Sarkar, Yeonju Ro, Sharath Nittur Sridhar, Zhangyang Wang, Aditya Akella, Souvik Kundu*  
> arXiv:2512.07173 | [PDF](https://arxiv.org/abs/2512.07173)

CadLLM is a training‑free, plug‑and‑play controller that improves the inference throughput of masked diffusion language models (dLLMs) by adapting decoding policies based on lightweight confidence signals produced by the model itself. Across GSM8K, MATH, MBPP and HumanEval, CadLLM delivers up to 2.28× throughput over strong Fast‑dLLM baselines while maintaining competitive accuracy.

<div align="center">
  <img src="asset/algo_overview.png" alt="CadLLM overview" width="600"/>
</div>

## Environment Setup

```bash
# Python 3.10+ recommended
pip install -r requirements.txt
```

You will also need access to [LLaDA](https://github.com/ML-GSAI/LLaDA) and [DREAM](https://github.com/DreamLM/Dream) model weights. You should not need to worry about downloading them manually as huggingface will automatically download the model when you run the scripts. However, if any issue arises, you can go to their github repo for more detailed download instructions.

## Usage

See ```eval.md``` in ```llada/``` and ```dream/``` for specific instructions.

## Citation

If you find this repository useful, please consider citing:

```bibtex
@misc{shen2026improvingthroughputdiffusionbasedlarge,
      title={Improving the Throughput of Diffusion-based Large Language Models via a Training-Free Confidence-Aware Calibration}, 
      author={Jucheng Shen and Gaurav Sarkar and Yeonju Ro and Sharath Nittur Sridhar and Zhangyang Wang and Aditya Akella and Souvik Kundu},
      year={2026},
      eprint={2512.07173},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2512.07173}, 
}
```