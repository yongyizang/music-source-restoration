#!/usr/bin/env python
import io, os, torch, numpy as np, soundfile as sf
from huggingface_hub import snapshot_download
from model import UFormer, UFormerConfig

# ——————————————————————
# 1) Setup
# ——————————————————————
REPO_ID  = "yongyizang/MSR_UFormers"
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
local_dir= snapshot_download(REPO_ID)
config   = UFormerConfig()
_model_cache = {}

VALID_CKPTS = [
    "acoustic_guitar","bass","electric_guitar","guitars","keyboards",
    "orchestra","rhythm_section","synth","vocals"
]

def _get_model(ckpt_name: str):
    if ckpt_name not in VALID_CKPTS:
        raise ValueError(f"Invalid checkpoint {ckpt_name!r}, choose from {VALID_CKPTS}")
    if ckpt_name in _model_cache:
        return _model_cache[ckpt_name]
    path = os.path.join(local_dir, "checkpoints", f"{ckpt_name}.pth")
    m = UFormer(config).to(device).eval()
    sd = torch.load(path, map_location="cpu")
    m.load_state_dict(sd)
    _model_cache[ckpt_name] = m
    return m

# ——————————————————————
# 2) Overlap-add helper
# ——————————————————————
def _overlap_add(model, x: np.ndarray, sr: int, chunk_s: float=5., hop_s: float=2.5):
    C, T = x.shape
    chunk, hop = int(sr*chunk_s), int(sr*hop_s)
    pad = (-(T - chunk) % hop) if T>chunk else 0
    x_pad = np.pad(x, ((0,0),(0,pad)), mode="reflect")
    win   = np.hanning(chunk)[None,:]
    out   = np.zeros_like(x_pad); norm = np.zeros((1,x_pad.shape[1]))
    n_chunks = 1 + (x_pad.shape[1] - chunk)//hop

    for i in range(n_chunks):
        s = i*hop
        seg = x_pad[:, s:s+chunk]
        seg = seg.astype(np.float32)  # Ensure float32 for model input
        with torch.no_grad():
            y = model(torch.from_numpy(seg[None]).to(device)).squeeze(0).cpu().numpy()
        out[:, s:s+chunk] += y * win
        norm[:, s:s+chunk] += win
    eps = 1e-8
    return (out / (norm + eps))[:, :T]

# ——————————————————————
# 3) HF Inference entry-point
# ——————————————————————
def inference(input_bytes: bytes, checkpoint: str = "guitars") -> bytes:
    """
    audio_bytes in → restored_bytes out.
    Pass {"inputs": <bytes>, "parameters": {"checkpoint": "<name>"}} to choose.
    """
    audio, sr = sf.read(io.BytesIO(input_bytes))
    if audio.ndim==1: audio = np.stack([audio,audio],axis=1)
    x = audio.T  # (C,T)

    model = _get_model(checkpoint)
    if x.shape[1] <= sr*5:
        with torch.no_grad():
            y = model(torch.from_numpy(x[None]).to(device)).squeeze(0).cpu().numpy()
    else:
        y = _overlap_add(model, x, sr)

    buf = io.BytesIO()
    sf.write(buf, y.T, sr, format="WAV")
    return buf.getvalue()

# ——————————————————————
# 4) CLI & Gradio
# ——————————————————————
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("UFormer RESTORE")
    parser.add_argument("-i","--input", type=str, help="noisy WAV")
    parser.add_argument("-o","--output",type=str, help="restored WAV")
    parser.add_argument("-c","--checkpoint",type=str,default="guitars",
                        choices=VALID_CKPTS)
    parser.add_argument("--serve",action="store_true", help="launch Gradio")
    args = parser.parse_args()

    if args.serve:
        import gradio as gr
        def _gr(path, ckpt):
            return inference(open(path,"rb").read(), checkpoint=ckpt)
        gr.Interface(
            fn=_gr,
            inputs=[
                gr.Audio(sources="upload", type="filepath"),
                gr.Dropdown(VALID_CKPTS, label="Checkpoint")
            ],
            outputs=gr.Audio(type="filepath"),
            title="🎵 Music Source Restoration Restoration",
            description="Choose which instrument/group model to run."
        ).launch()

    else:
        assert args.input and args.output
        out = inference(open(args.input,"rb").read(),
                        checkpoint=args.checkpoint)
        open(args.output,"wb").write(out)
        print(f"✅ Restored → {args.output} using {args.checkpoint}")
