# Music Source Restoration
Official Repository for "Music Source Restoration"

[Paper Link](./preprint.pdf)[![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-sm.svg)](https://huggingface.co/datasets/yongyizang/RawStems)[![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/models)[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/yongyizang/MusicSourceRestoration)


Note that the main purpose of this repository is a starting template for the research/engineering community to implement their own models. It is not intended to be a production-ready solution.
The model we implemented is a baseline model, and is **not expected to perform well** in production use.

## Getting Started
```bash
git clone https://github.com/yongyizang/music-source-restoration
cd music-source-restoration
uv venv msr
source msr/bin/activate
pip install -r requirements.txt
```

## Inference Using Pretrained Models
For more details, please refer to `inference.py`. 50% Overlap-Add is used by default.

**IMPORTANT: Please note that these pre-trained models are baseline models. They are likely not going to perform well for production use. Please wait for research/engineering community implementations!**

```bash
python inference.py -i {input_file} -o {output_file} -c {checkpoint_file}
```

Alternatively, you could serve the model using Gradio:
```bash
python inference.py --serve
```

## Training
```bash
python train.py -c {config_file}
```

Please see example config files in `configs/`.