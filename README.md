# SoundStream Neural Audio Codec

Here's my implementation of [SoundStream](https://arxiv.org/abs/2107.03312) for 6 kbps speech compression on LibriSpeech.

## Results (test-clean, full audio)

| Metric | Value     | Required |
|--------|-----------|----------|
| STOI   | **0.897** | > 0.80   |
| NISQA  | **2.883** | > 2.25   |

Training logs with validation: [Comet ML](https://www.comet.com/alex-plotnikov/bhw-soundstream-plotnikov244/)

Pretrained model: [HuggingFace](https://huggingface.co/AlexPlotnikovTech/soundstream-libri)

## Demo

See [`demo.ipynb`](demo.ipynb) — open in Google Colab, provide an audio URL, run all cells.

## Installation

\```bash
git clone https://github.com/AlexanderPlotnikovv/SoundStreamHW.git
\``` \
\```
cd SoundStreamHW
\``` \
\```
pip install -r requirements.txt
\```

## Download pre-trained checkpoint

\```python
from huggingface_hub import snapshot_download
\``` \
\```
snapshot_download(
    repo_id="AlexPlotnikovTech/soundstream-libri",
    local_dir="soundstream_libri",
)
\```

## Training

The model was trained on LibriSpeech `train-clean-100`. To reproduce:

1. Set `COMET_API_KEY` environment variable
2. Place LibriSpeech at `data/LibriSpeech/` (or use `download=True` in dataset config)
3. Run:

\```bash
python3 train.py -cn=soundstream trainer.device=cuda
\```

Training takes ~4 hours on A100 (50 epochs).

## Evaluation

Computes STOI on full test-clean audio and saves reconstructions for NISQA:

\```bash
python3 inference.py inferencer.device=cuda dataloader.batch_size=1
\```

For NISQA, run [NISQA](https://github.com/gabrielmittag/NISQA) on the saved `fake_*.wav` files.

## Architecture

- **Encoder**: 4 EncoderBlocks, strides [2, 4, 5, 5], output dim 128
- **Quantizer**: Residual Vector Quantizer, 8 layers, 1024 codes each, EMA updates
- **Decoder**: mirror of encoder
- **Discriminators**: WaveDiscriminator (3 scales) + STFTDiscriminator

## References

Zeghidour et al., "SoundStream: An End-to-End Neural Audio Codec",
2021. [arXiv:2107.03312](https://arxiv.org/abs/2107.03312)
