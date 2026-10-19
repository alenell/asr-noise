# Noise-Robust Self-Supervised ASR

Research project comparing techniques for improving Wav2Vec2 speech recognition at low signal-to-noise ratios (SNR). Accompanies the paper **"Comparative Analysis on Self-Supervised Automatic Speech Recognition Techniques for Low Signal-to-Noise Ratios"** (Alex Lenell, Johns Hopkins University, Whiting School of Engineering).

## Overview

Self-supervised ASR models such as Wav2Vec2 degrade sharply on noisy audio. This project fine-tunes [`facebook/wav2vec2-base-960h`](https://huggingface.co/facebook/wav2vec2-base-960h) and measures word error rate (WER) on LibriSpeech test-clean, clean and with noise added at -5, 0, 5 and 10 dB SNR, to find which of the following helps most:

- **Noise-augmented fine-tuning**: LibriSpeech dev-clean mixed with WHAM! environmental noise
- **Adversarial fine-tuning**: FGSM perturbations (ε = 0.001) that increase the CTC loss
- **Noise + adversarial fine-tuning**: FGSM examples generated from the noise-augmented data
- **LLM rescoring**: GPT-2 rescoring of the top-5 CTC beam-search candidates, `S = S_CTC + α·S_LM` with α = 0.1

Each technique is evaluated with and without rescoring, giving eight configurations in total.

## Repository contents

| File | Description |
|------|-------------|
| `ResearchProject.ipynb` | All code: data loading, noise augmentation, baseline evaluation, LLM rescoring, fine-tuning, adversarial training |
| `Self_Supervised_Automatic_Speech_Recognition.pdf` | The paper |

## Results

WER (%) on LibriSpeech test-clean, clean and with WHAM! noise added. Fine-tuning was 1 epoch.

| SNR   | Baseline | + LM  | Noise FT | + LM  | Adv. FT | + LM  | Noise + Adv. FT | + LM  |
|-------|---------:|------:|---------:|------:|--------:|------:|----------------:|------:|
| Clean | 17.02    | 16.75 | 5.19     | 4.92  | 4.65    | 4.26  | 5.67            | 5.43  |
| -5 dB | 84.99    | 87.71 | 68.41    | 69.45 | 77.04   | 78.24 | 69.58           | 70.07 |
| 0 dB  | 66.84    | 69.29 | 41.09    | 42.81 | 51.05   | 50.59 | 42.16           | 43.54 |
| 5 dB  | 43.90    | 47.38 | 19.54    | 20.67 | 23.43   | 24.13 | 19.93           | 20.74 |
| 10 dB | 28.96    | 30.61 | 10.15    | 10.15 | 10.97   | 10.86 | 10.59           | 10.63 |

**Takeaways:** noise-augmented fine-tuning performs best in noisy conditions; adversarial fine-tuning gives the best clean-speech WER; LLM rescoring helps little and often hurts at low SNR; combining all techniques did not beat noise augmentation alone. See the paper for details and limitations.

## Setup

### Requirements

- Python 3.10+
- A CUDA GPU is strongly recommended for fine-tuning and evaluation

```bash
pip install torch torchaudio transformers pandas numpy jiwer pyctcdecode jupyter
```

### Data

Download the following and note where you put them:

1. **LibriSpeech** `dev-clean` and `test-clean` from [OpenSLR](https://www.openslr.org/12)
2. **WHAM! noise** (`tr` audio folder and `noise_meta_tr.csv`) from [wham.whisper.ai](https://wham.whisper.ai/)

### Configure paths

The notebook currently contains hardcoded local Windows paths. Before running, update these to match your machine:

- The WHAM! metadata CSV (`noise_meta_tr.csv`) and noise root (`tr`) in the **SNR Files** cell
- `root` (dev-clean) and `test_root` (test-clean) in the **Load Data** cell
- The path to the noise-tuned checkpoint in the commented-out model-loading cell

## Running the notebook

Run the cells in order. The notebook is organised into these sections:

1. **SNR Augmentation**: `mix_with_noise` mixes a noise clip into speech at a target SNR
2. **SNR Files / Load Data**: selects one WHAM! noise file per location ID (seed 42) and loads LibriSpeech transcripts
3. **Datasets / Collate Functions**: `SpeechDataset` with optional on-the-fly noise augmentation, plus train and eval collators
4. **Word Error Rate Baseline**: evaluates the pretrained model on clean and noisy test sets
5. **LLM Rescoring**: pyctcdecode beam search with GPT-2 scoring
6. **Fine-Tuning With Noisy Data**: noise-augmented fine-tuning
7. **Adversarial Training**: FGSM attack and adversarial fine-tuning
8. **Noisy + Adversarial**: both combined

Most training and evaluation cells are **commented out**, because the experiments are slow and the original outputs are recorded beneath each cell. To reproduce a result, uncomment the relevant cell(s). Each fine-tuning section reloads the pretrained model, trains, and saves a checkpoint (`./fine_tuned_model`, `./adversarial_fine_tuned_model`, `./noisy_adversarial_fine_tuned_model`).

### Experiment settings

| Setting | Value |
|---------|-------|
| Base model | `facebook/wav2vec2-base-960h` |
| Fine-tuning data | LibriSpeech dev-clean (2,703 utterances, ~5.4 h) |
| Test data | LibriSpeech test-clean (2,620 utterances) |
| Noise | WHAM!, one file per location, SNRs of -5, 0, 5, 10 dB |
| Optimiser | AdamW, lr 1e-5, batch size 4, 1 epoch, feature encoder frozen |
| FGSM ε | 0.001 |
| Rescoring | GPT-2, CTC beam width 10, top 5 candidates, α = 0.1 |
| Eval batch size | 8 |

## Notes and limitations

- Fine-tuning ran for a single epoch, so the models may not have fully converged.
- ε and α were not extensively tuned (α was checked on 200 dev utterances over 0.0–0.3).
- Only GPT-2 was tried for rescoring, and no denoising front end was tested.
- The test-set noise is sampled with a shared random generator, so re-running may select different noise clips than the original run and give slightly different WERs.

## Citation

```bibtex
@misc{lenell2026noiserobust,
  author = {Lenell, Alex},
  title  = {Comparative Analysis on Self-Supervised Automatic Speech Recognition Techniques for Low Signal-to-Noise Ratios},
  school = {Johns Hopkins University, Whiting School of Engineering}
}
```
