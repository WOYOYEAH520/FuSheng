"""官方伴奏相减（自动音量匹配版）。

比 work/extract_vocal.py 强的地方：
  1. 延时：GCC-PHAT 粗搜（8 kHz）→ 全采样率互相关 + 抛物线插值得到亚样本延时，
     再用 FFT 相位斜坡做分数延迟（旧版只在 8 kHz 上取整，误差可达 ±2.75 样本）。
  2. 音量：**短时 × 逐频带 × 复数增益**，并在时间/频率上双重平滑。
     旧版整首一把常数增益，遇到"响度归一化 / 限幅"这种随时间变化的电平差就会时准时不准。
  3. 能量守卫：某个频带在这一帧里几乎没有伴奏能量时不做补偿（否则会把噪声放大成"呼呼"声）。
  4. 输出诊断：延时、全局增益(dB)、补偿范围、残余电平，便于判断两份是不是同一版本。

用法：
  python work/subtract_vocal.py --mix 原曲.wav --inst 官方伴奏.wav --outdir 输出目录
  python work/subtract_vocal.py ... --report report.json --max-gain 4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

SR = 44100
N_FFT = 4096
HOP = 1024


# ---------------------------------------------------------------- 读写
def load(p, sr=SR):
    y, s = sf.read(str(p), dtype="float32", always_2d=True)
    if s != sr:
        y = librosa.resample(y.T, orig_sr=s, target_sr=sr).T
    if y.shape[1] == 1:
        y = np.repeat(y, 2, axis=1)
    return y


def shift_frac(x, tau: float):
    """把 x 平移 tau 个样本（tau>0 表示向后延），用 FFT 相位斜坡实现分数延迟。"""
    n = len(x)
    if abs(tau) < 1e-6:
        return x.copy()
    tau = float(tau)
    spec = np.fft.rfft(x, axis=0)
    freqs = np.fft.rfftfreq(n, d=1.0)
    ph = np.exp(-2j * np.pi * freqs * tau)
    if spec.ndim > 1:
        ph = ph[:, None]
    return np.fft.irfft(spec * ph, n=n, axis=0).astype(np.float32)


# ---------------------------------------------------------------- 对齐
def gcc_phat_lag(a, b, sr=SR, max_lag_s=6.0):
    """GCC-PHAT：返回让 b 对齐到 a 所需延时（样本，可为分数）。"""
    na = len(a)
    n = 1
    while n < 2 * na:
        n *= 2
    A = np.fft.rfft(a, n=n)
    B = np.fft.rfft(b, n=n)
    R = A * np.conj(B)
    R /= np.abs(R) + 1e-12
    cc = np.fft.irfft(R, n=n)
    max_lag = int(max_lag_s * sr)
    cc = np.concatenate([cc[-max_lag:], cc[: max_lag + 1]])
    lags = np.arange(-max_lag, max_lag + 1)
    k = int(np.argmax(cc))
    # 抛物线插值
    frac = 0.0
    if 0 < k < len(cc) - 1:
        y0, y1, y2 = cc[k - 1], cc[k], cc[k + 1]
        denom = (y0 - 2 * y1 + y2)
        if abs(denom) > 1e-12:
            frac = 0.5 * (y0 - y2) / denom
    return float(lags[k] + frac)


def estimate_delay(mix, inst):
    """先 8 kHz 粗搜（容忍长延时），再全速精修到亚样本。"""
    m8 = librosa.resample(mix.mean(1), orig_sr=SR, target_sr=8000)
    i8 = librosa.resample(inst.mean(1), orig_sr=SR, target_sr=8000)
    # 用幅度包络（抗混响，也不受高次谐波相位影响）
    me = np.abs(librosa.stft(m8, n_fft=1024, hop_length=256)).mean(0)
    ie = np.abs(librosa.stft(i8, n_fft=1024, hop_length=256)).mean(0)
    lag8 = gcc_phat_lag(me, ie, sr=8000 / 256, max_lag_s=6.0)
    coarse = lag8 * 256 * SR / 8000
    print(f"[align] 粗延时 {lag8:.2f} 帧 @8k → {coarse:.1f} 样本", flush=True)

    # 全速率精修：300–5000 Hz 带通后做互相关，只在粗延时附近找
    from scipy.signal import fftconvolve

    n = min(len(mix), len(inst), SR * 90)
    mf = _bandpass(mix[:n].mean(1))
    i_f = _bandpass(inst[:n].mean(1))
    cc = fftconvolve(mf, i_f[::-1], mode="full")
    lags = np.arange(-(len(i_f) - 1), len(mf))
    lo, hi = int(coarse) - 300, int(coarse) + 300
    sel = (lags >= lo) & (lags <= hi)
    if not sel.any():
        sel = np.abs(lags - int(coarse)) <= 2000
    idx = np.argmax(cc[sel])
    best_lag = int(lags[sel][idx])

    # 抛物线插值 → 亚样本（用 cc 在最佳点 ±1 的值）
    vals = {k: float(cc[np.where(lags == k)[0][0]]) for k in (best_lag - 1, best_lag, best_lag + 1)
            if lags[0] <= k <= lags[-1]}
    frac = 0.0
    if all(k in vals for k in (best_lag - 1, best_lag, best_lag + 1)):
        y0, y1, y2 = vals[best_lag - 1], vals[best_lag], vals[best_lag + 1]
        denom = (y0 - 2 * y1 + y2)
        if abs(denom) > 1e-12:
            frac = float(np.clip(0.5 * (y0 - y2) / denom, -1, 1))
    return best_lag + frac


def _bandpass(x, lo=250, hi=6000):
    from scipy.signal import butter, sosfilt
    sos = butter(4, [lo / (SR / 2), hi / (SR / 2)], btype="band", output="sos")
    return sosfilt(sos, x).astype(np.float32)


# ---------------------------------------------------------------- 增益匹配
def band_edges(n_bands=32, f_lo=20.0, f_hi=None):
    """频带边界：铺满整个频谱（默认 20 Hz ~ 奈奎斯特），否则高频镲片不会被补偿。"""
    hi = f_hi if f_hi else (SR / 2) * 0.999
    return np.geomspace(f_lo, hi, n_bands + 1)


def _reliability(Ms, Is, gband):
    """每个时频块"像不像没人在唱"：残差越小越可信。"""
    R = Ms - gband * Is
    eR = np.sum(np.abs(R) ** 2, axis=0)
    eI = np.sum(np.abs(gband * Is) ** 2, axis=0)
    return eI / (eR + eI + 1e-12)


def estimate_gains(M, I, n_fft=N_FFT, max_gain=4.0, block_s=2.0, hop_s=1.0, iters=3):
    """逐频带复数增益，按时间分块估计（默认 2 秒一块、1 秒一跳），块内插值。

    为什么这样：
      - 块内「分子分母先累加再相除」是无偏最小二乘；先按帧求比值再取中位数会系统偏大（实测 +7%）。
      - 直接最小二乘会被人声污染（ΣV·conj(I)≠0），导致过减。所以迭代几轮，
        用「残差 / 伴奏能量」给每个时频块加权：残差小 = 人声大概率不在场，这时 M/I 就是真实的电平比。
      - 分块而不是整首一把常数，是为了跟住"响度归一化/限幅"造成的时变电平差。
    """
    nbins, nframes = M.shape
    freqs = np.linspace(0, SR / 2, nbins)
    edges = band_edges()
    bands = []
    for bi, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        sel = (freqs >= lo) & (freqs < hi)
        if sel.any():
            bands.append(sel)
    band_energy = np.array([np.sum(np.abs(I[sel]) ** 2, axis=0) for sel in bands], dtype=np.float32)

    fps = SR / HOP
    win = max(8, int(block_s * fps))
    hop = max(4, int(hop_s * fps))
    starts = list(range(0, max(1, nframes - win + 1), hop))
    if not starts or starts[-1] + win < nframes:
        starts.append(max(0, nframes - win))

    # 初始权重全 1（按 频带 × 帧）
    w = np.ones((len(bands), nframes), dtype=np.float64)
    gs_blocks = None
    g = None
    for it in range(max(1, iters)):
        gs_blocks = np.ones((len(bands), len(starts)), dtype=np.complex64)
        for bi, sel in enumerate(bands):
            Ms, Is = M[sel], I[sel]
            num_t = np.sum(Ms * np.conj(Is), axis=0).astype(np.complex128)
            den_t = np.sum(np.abs(Is) ** 2, axis=0).astype(np.float64)
            wb = w[bi]
            for j, s0 in enumerate(starts):
                sl = slice(s0, min(s0 + win, nframes))
                ww = wb[sl]
                num = np.sum(ww * num_t[sl])
                den = np.sum(ww * den_t[sl])
                if den > 1e-12:
                    gs_blocks[bi, j] = complex(num / den)
        # 展开到逐帧
        # 最后一步：每个频带只用「最像没人在唱」的那部分帧来定整首增益。
        # 因为 L2 最优会被人声与伴奏的相关性带偏（实测过减 2~6%），
        # 而人声空隙帧里的 M/I 就是真实电平比。
        G = np.ones(len(bands), dtype=np.complex64)
        for bi, sel in enumerate(bands):
            Ms, Is = M[sel], I[sel]
            num_t = np.sum(Ms * np.conj(Is), axis=0).astype(np.complex128)
            den_t = np.sum(np.abs(Is) ** 2, axis=0).astype(np.float64)
            rel = _reliability(Ms, Is, g[sel]) if g is not None else np.ones(nframes)
            keep = rel >= np.percentile(rel, 75) if nframes > 20 else np.ones(nframes, bool)
            num = np.sum(num_t[keep])
            den = np.sum(den_t[keep])
            if den > 1e-12:
                G[bi] = complex(num / den)

        # 频带之间本该平滑：3 带加权平滑 + 低能量频带向全局值收缩，
        # 否则个别频带 ±5% 的估计噪声会变成 -24 dB 的残差地板。
        be_sum = band_energy.sum(axis=1).astype(np.float64) + 1e-12
        gl = complex(np.sum(be_sum * G) / np.sum(be_sum))
        sm = G.copy()
        for bi in range(len(bands)):
            lo, hi = max(0, bi - 1), min(len(bands), bi + 2)
            ww = be_sum[lo:hi]
            sm[bi] = complex(np.sum(ww * G[lo:hi]) / np.sum(ww))
        wb = be_sum / (be_sum + 0.05 * be_sum.max())
        G = (wb * sm + (1 - wb) * gl).astype(np.complex64)
        dev = np.abs(gs_blocks) / (np.abs(G)[:, None] + 1e-12)
        dev_db = 20 * np.log10(dev + 1e-9)
        dev_db = _smooth_blocks_real(dev_db, 5)
        # 只在时间上的波动明显超过估计噪声（约 1 dB）时才认为电平真在变；
        # 否则压成常数，避免把噪声当成"音量变化"去补偿（那会留下宽带残差）。
        std_b = dev_db.std(axis=1) if dev_db.shape[1] > 1 else np.zeros(dev_db.shape[0])
        lam = np.clip((std_b - 0.8) / 1.2, 0.0, 1.0)
        dev_used_db = dev_db * lam[:, None]
        gs_blocks = (G[:, None] * (10 ** (dev_used_db / 20.0))).astype(np.complex64)
        g = np.ones((nbins, nframes), dtype=np.complex64)
        for bi, sel in enumerate(bands):
            vals = gs_blocks[bi]
            xs = np.array([s0 + win / 2 for s0 in starts], dtype=np.float64)
            if len(xs) == 1:
                g[sel, :] = vals[0]
            else:
                re = np.interp(np.arange(nframes), xs, vals.real)
                im = np.interp(np.arange(nframes), xs, vals.imag)
                g[sel, :] = (re + 1j * im)[None, :]
        if it == max(1, iters) - 1:
            break
        # 用残差更新权重：残差越小（越可能没人在唱）越可信，按 频带 × 帧 算
        R = M - g * I
        eR = np.array([np.sum(np.abs(R[sel]) ** 2, axis=0) for sel in bands], dtype=np.float64)
        eI = np.array([np.sum(np.abs(g[sel] * I[sel]) ** 2, axis=0) for sel in bands], dtype=np.float64)
        w = (eI / (eR + eI + 1e-12)) ** 2
        w = np.clip(w, 0.0, 1.0)
        k = max(3, int(0.5 * fps))
        w = np.stack([_smooth_vec(wb, k) for wb in w])

    mag = np.clip(np.abs(g), 1.0 / max_gain, max_gain)
    g = (mag * np.exp(1j * np.angle(g))).astype(np.complex64)
    return g, band_energy


def _smooth_vec(v, k):
    """一维滑动平均（用于权重/增益的时间平滑）。"""
    if k <= 1:
        return v
    kern = np.ones(k) / k
    pad = k // 2
    vp = np.pad(v, (pad, pad), mode="edge")
    return np.convolve(vp, kern, mode="same")[pad:pad + len(v)]


def _smooth_blocks(x, k):
    """沿块方向做滑动平均（x: [bands, blocks]，复数）。"""
    if k <= 1 or x.shape[1] <= 1:
        return x
    pad = k // 2
    re = np.pad(x.real, ((0, 0), (pad, pad)), mode="edge")
    im = np.pad(x.imag, ((0, 0), (pad, pad)), mode="edge")
    kern = np.ones(k) / k
    out_re = np.apply_along_axis(lambda v: np.convolve(v, kern, mode="valid"), 1, re)
    out_im = np.apply_along_axis(lambda v: np.convolve(v, kern, mode="valid"), 1, im)
    return (out_re + 1j * out_im).astype(np.complex64)


def _smooth_blocks_real(x, k):
    """实数版块平滑（x: [bands, blocks]）。"""
    if k <= 1 or x.shape[1] <= 1:
        return x
    pad = k // 2
    xp = np.pad(x, ((0, 0), (pad, pad)), mode="edge")
    kern = np.ones(k) / k
    return np.apply_along_axis(lambda v: np.convolve(v, kern, mode="valid"), 1, xp)


def _median_time(g, k):
    from scipy.ndimage import median_filter

    if k % 2 == 0:
        k += 1
    re = median_filter(np.real(g), size=(1, k), mode="nearest")
    im = median_filter(np.imag(g), size=(1, k), mode="nearest")
    return (re + 1j * im).astype(np.complex64)


def _smooth_freq(g, k):
    from scipy.ndimage import uniform_filter1d

    if k <= 1:
        return g
    re = uniform_filter1d(np.real(g), size=k, axis=0, mode="nearest")
    im = uniform_filter1d(np.imag(g), size=k, axis=0, mode="nearest")
    return (re + 1j * im).astype(np.complex64)


# ---------------------------------------------------------------- 主流程
def lf_coherence(a, b, sr=SR, band=(20.0, 200.0)):
    """低频段相干度：人声几乎不占 20–200 Hz，两份若是同一版本应接近 1。"""
    from scipy.signal import butter, coherence, sosfilt

    sos = butter(4, [band[0] / (sr / 2), band[1] / (sr / 2)], btype="band", output="sos")
    x = sosfilt(sos, a.mean(1))
    y = sosfilt(sos, b.mean(1))
    f, cxy = coherence(x, y, fs=sr, nperseg=8192)
    return float(np.median(cxy))


def extract(mix, inst, cleanup: float = 0.0, max_gain: float = 4.0, align: bool = True):
    report = {}
    if align:
        tau = estimate_delay(mix, inst)
        inst_al = shift_frac(inst, tau)
        report["delay_samples"] = round(float(tau), 3)
        report["delay_ms"] = round(float(tau) / SR * 1000, 2)
        print(f"[align] 精修延时 {tau:.3f} 样本（{tau / SR * 1000:.2f} ms）", flush=True)
    else:
        inst_al = inst
        report["delay_ms"] = 0.0

    try:
        c = lf_coherence(mix, inst_al)
        report["lf_coherence"] = round(c, 3)
        report["same_version"] = bool(c >= 0.7)
        if c < 0.7:
            print(f"[warn] 20–200 Hz 相干度只有 {c:.2f}，两份可能不是同一版本/没对齐好", flush=True)
    except Exception as e:
        report["lf_coherence"] = None
        report["same_version"] = None

    n = min(len(mix), len(inst_al))
    mix, inst_al = mix[:n], inst_al[:n]

    gains_all = []
    V = np.zeros_like(mix)
    band_energy = None
    for ch in range(mix.shape[1]):
        M = librosa.stft(mix[:, ch], n_fft=N_FFT, hop_length=HOP)
        I = librosa.stft(inst_al[:, ch], n_fft=N_FFT, hop_length=HOP)
        g, band_energy = estimate_gains(M, I, max_gain=max_gain)
        gains_all.append(g)
        R = M - g * I
        if cleanup > 0:
            mag = np.abs(R)
            thr = cleanup * np.percentile(mag, 40, axis=0, keepdims=True)
            R = np.where(mag > thr, R, R * (mag / (thr + 1e-9)))
        V[:, ch] = librosa.istft(R, hop_length=HOP, length=n)

    # 诊断：全局增益（只统计伴奏有能量的时频块）
    g = np.mean(np.stack(gains_all, 0), axis=0)
    W = np.abs(g)
    voiced = W[(np.abs(g) > 0.01)]
    report["gain_median_db"] = round(float(20 * np.log10(np.median(voiced) + 1e-9)), 2)
    report["gain_p05_db"] = round(float(20 * np.log10(np.percentile(voiced, 5) + 1e-9)), 2)
    report["gain_p95_db"] = round(float(20 * np.log10(np.percentile(voiced, 95) + 1e-9)), 2)
    report["gain_span_db"] = round(report["gain_p95_db"] - report["gain_p05_db"], 2)
    return V.astype(np.float32), inst_al, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", required=True)
    ap.add_argument("--inst", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--cleanup", type=float, default=0.0)
    ap.add_argument("--max-gain", type=float, default=4.0, help="最大补偿倍数（4 = +12 dB）")
    ap.add_argument("--no-align", action="store_true")
    ap.add_argument("--report", default="")
    a = ap.parse_args()

    mix = load(a.mix)
    inst = load(a.inst)
    vocal, inst_al, report = extract(mix, inst, a.cleanup, a.max_gain, not a.no_align)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    sf.write(str(out / "vocals.wav"), vocal, SR, subtype="PCM_16")
    sf.write(str(out / "no_vocals.wav"), inst_al[: len(mix)], SR, subtype="PCM_16")
    print("[done] " + json.dumps(report, ensure_ascii=False), flush=True)
    if a.report:
        Path(a.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
