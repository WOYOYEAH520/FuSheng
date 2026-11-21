"""BS-RoFormer 人声分离（pymss 后端）

用法:
  .venv\\Scripts\\python.exe tools\\sep_roformer.py --input song.wav --outdir work\\sep
输出: <outdir>\\vocals.wav, <outdir>\\no_vocals.wav （44.1k 立体声）
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "tools" / "pymss_weights"
MODEL = WEIGHTS / "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
CONFIG = WEIGHTS / "model_bs_roformer_ep_317_sdr_12.9755.yaml"

AUDIO_PARAMS = {
    "wav_bit_depth": "FLOAT", "flac_bit_depth": "PCM_24",
    "mp3_bit_rate": "320k", "m4a_bit_rate": "320k",
    "m4a_codec": "aac", "m4a_aac_at_quality": 2,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--model", default=str(MODEL))
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    from pymss import MSSeparator, load_audio

    print("loading BS-RoFormer ...", flush=True)
    sep = MSSeparator(
        model_type="bs_roformer",
        model_path=a.model,
        config_path=a.config,
        device=a.device,
        device_ids=[0],
        output_format="wav",
        use_tta=False,
        store_dirs={},
        audio_params=AUDIO_PARAMS,
        logger=None,
        debug=False,
        inference_params={
            "batch_size": 1,
            "chunk_size": 352800,
            "overlap_size": 264600,
            "standardize": False,
            "normalize": False,
            "use_amp": True,
            "model_dtype": "float16",
            "cuda_attention_backend": "default",
        },
    )
    print("separating ...", flush=True)
    mix, sr = load_audio(a.input, sr=44100, mono=False)
    res = sep.separate(mix, pbar=False)
    print("stems:", list(res.keys()), flush=True)

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    def pick(*names):
        for n in names:
            if n in res:
                return np.asarray(res[n], dtype=np.float32)
        raise KeyError(f"找不到 {names}，实际是 {list(res)}")

    voc = pick("vocals")
    ins = pick("instrumental", "other")
    print("shapes:", voc.shape, ins.shape, flush=True)

    def orient(arr):
        if arr.ndim == 1:
            return arr
        return arr.T if arr.shape[0] <= arr.shape[1] else arr

    for name, arr in (("vocals.wav", voc), ("no_vocals.wav", ins)):
        data = orient(arr)
        sf.write(str(out / name), np.ascontiguousarray(data, dtype=np.float32), 44100, subtype="PCM_16")
    print(f"done -> {out}/vocals.wav, no_vocals.wav", flush=True)


if __name__ == "__main__":
    main()
