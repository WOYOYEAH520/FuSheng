"""
本地翻唱工作台 —— 分步式：源音频 → 官方伴奏 → 分离 → 换音色 → 混音母带
                    以及：用自己的素材训练新的音色模型

启动:  python webapp\\app.py    浏览器打开 http://127.0.0.1:7860
"""
from __future__ import annotations

import json
import math
import mimetypes
import os
import re
import shutil
import subprocess
import threading
import time
import traceback
import sys
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
import uvicorn

# ---------------------------------------------------------------- 路径与配置
BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
BIN_DIR = ROOT / "tools" / "bin"


def _venv_python() -> str:
    """有 .venv 就用它；没有（例如用别的虚拟环境跑）就用当前解释器。"""
    cand = ROOT / ".venv" / "Scripts" / "python.exe"
    if os.name != "nt":
        cand = ROOT / ".venv" / "bin" / "python"
    return str(cand) if cand.exists() else sys.executable


def _ffmpeg() -> str:
    """优先用自带的 ffmpeg，其次用 PATH 里的。"""
    cand = BIN_DIR / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if cand.exists():
        return str(cand)
    return shutil.which("ffmpeg") or "ffmpeg"


PY = _venv_python()
FFMPEG = _ffmpeg()
RVC_DIR = ROOT / "tools" / "rvc_src" / "Retrieval-based-Voice-Conversion-WebUI-main"
WEIGHTS = RVC_DIR / "assets" / "weights"
INDICES = RVC_DIR / "assets" / "indices"
LOGS = RVC_DIR / "logs"
REGISTRY = BASE / "models.json"
JOBS = BASE / "jobs"          # 训练任务的临时目录
SESSIONS = BASE / "sessions"  # 翻唱工作台：每一步的产物都留在这里
JOBS.mkdir(exist_ok=True)
SESSIONS.mkdir(exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)  # 训练/索引目录，首次运行可能还没有

# 子进程统一走 UTF-8，避免中文日志在控制台/管道里变乱码
CHILD_ENV = {"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}

AUDIO_EXT = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".opus", ".wma", ".aiff", ".webm"}
NAME_RE = re.compile(r"^[A-Za-z0-9_\-]{2,32}$")
SID_RE = re.compile(r"^[0-9a-f]{10}$")
BILI_RE = re.compile(r"(BV[0-9A-Za-z]{10})")
PITCH_RE = re.compile(r"^[+-]?\d{1,2}$")

app = FastAPI(title="浮声 翻唱工作台")

_sessions: dict[str, dict] = {}
_sessions_lock = threading.RLock()
_active_sids: set[str] = set()

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_gpu_lock = threading.Lock()
_cancel: dict[str, threading.Event] = {}
_registry_lock = threading.RLock()


# ================================================================ 会话存储
def _write_session(s: dict):
    d = SESSIONS / s["id"]
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "session.json.tmp"
    tmp.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, d / "session.json")


def get_session(sid: str) -> dict:
    """内存优先；重启后从磁盘恢复。"""
    if not SID_RE.match(sid or ""):
        raise HTTPException(404, "任务不存在")
    with _sessions_lock:
        s = _sessions.get(sid)
        if s is not None:
            return s
        p = SESSIONS / sid / "session.json"
        if not p.exists():
            raise HTTPException(404, "任务不存在")
        try:
            s = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            raise HTTPException(500, "任务数据损坏")
        job = dict(s.get("job") or {})
        # 服务重启后，内存里没有活动线程，把“进行中”纠正为中断
        if job.get("status") == "running" and sid not in _active_sids:
            job.update(status="error", progress=0,
                       message="服务重启，这一步被中断了，请重新执行")
            s["job"] = job
            s["updated"] = time.time()
            _write_session(s)
        _sessions[sid] = s
        return s


def patch_session(sid: str, changes: Optional[dict] = None, drop: tuple = ()) -> dict:
    with _sessions_lock:
        s = get_session(sid)
        if changes:
            s.update(changes)
        for k in drop:
            s.pop(k, None)
        s["updated"] = time.time()
        _write_session(s)
        return s


def set_job(sid: str, **kw):
    with _sessions_lock:
        s = get_session(sid)
        job = dict(s.get("job") or {})
        job.update(kw)
        s["job"] = job
        s["updated"] = time.time()
        _write_session(s)


def progress(sid: str, pct: int, message: str):
    set_job(sid, progress=max(0, min(99, int(pct))), message=message)


def fail_stage(sid: str, err: Exception):
    try:
        with _sessions_lock:
            s = get_session(sid)
            job = dict(s.get("job") or {})
            job.update(status="error", message=str(err)[:800] or "未知错误")
            s["job"] = job
            s["updated"] = time.time()
            try:
                (SESSIONS / sid / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
            except Exception:
                pass
            _write_session(s)
    except Exception:
        pass


def ensure_idle(sid: str):
    s = get_session(sid)
    if (s.get("job") or {}).get("status") == "running":
        raise HTTPException(409, "已有步骤正在处理，请等它完成")


def claim_job(sid: str, key: str, message: str):
    with _sessions_lock:
        s = get_session(sid)
        if (s.get("job") or {}).get("status") == "running":
            raise HTTPException(409, "已有步骤正在处理，请等它完成")
        s["job"] = {"key": key, "status": "running", "progress": 1,
                    "message": message, "started": time.time()}
        s["updated"] = time.time()
        _write_session(s)


def start_stage(sid: str, fn, args: tuple):
    def runner():
        _active_sids.add(sid)
        try:
            fn(sid, *args)
        except Exception as e:  # 兜底，避免会话永远停在“进行中”
            fail_stage(sid, e)
        finally:
            _active_sids.discard(sid)

    threading.Thread(target=runner, daemon=True).start()


# ================================================================ 会话文件
def _rm(paths):
    for p in paths:
        try:
            if p.is_dir():
                shutil.rmtree(p)
            elif p.exists():
                p.unlink()
        except Exception:
            pass


def _d(s: dict) -> Path:
    return SESSIONS / s["id"]


def wipe_inst_files(s: dict):
    d = _d(s)
    _rm(sorted(d.glob("inst_raw*")) + sorted(d.glob("inst_upload*")) + [d / "inst_source.wav"])


def wipe_sep_files(s: dict):
    d = _d(s)
    _rm([d / "roformer", d / "extracted"])


def wipe_conv_files(s: dict):
    d = _d(s)
    _rm([d / "vocal_rvc.wav"])


def wipe_mix_files(s: dict):
    d = _d(s)
    _rm([d / "final.wav", d / "final.mp3", d / "mix_raw.wav"])


def clear_inst(sid: str):
    """清除官方伴奏，同时作废依赖它的分离 / 换声 / 成品。"""
    with _sessions_lock:
        s = get_session(sid)
        patch_session(sid, drop=("inst", "sep", "conv", "mix"))
        wipe_inst_files(s)
        wipe_sep_files(s)
        wipe_conv_files(s)
        wipe_mix_files(s)


# ================================================================ 小工具
def fmt_dur(sec) -> str:
    if not sec:
        return ""
    sec = int(round(float(sec)))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def audio_duration(p: Path) -> Optional[float]:
    try:
        import soundfile as sf
        info = sf.info(str(p))
        if info.samplerate:
            return round(info.frames / info.samplerate, 2)
    except Exception:
        pass
    return None


def safe_filename(text: str, fallback: str = "翻唱") -> str:
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(text or "")).strip().strip(".")
    return (text[:60] or fallback)


def is_bilibili(text: str) -> bool:
    text = (text or "").strip()
    return "bilibili.com" in text or bool(re.fullmatch(r"(BV[0-9A-Za-z]{10}|av\d+)", text))


def bilibili_bvid(text: str) -> Optional[str]:
    m = BILI_RE.search(text or "")
    return m.group(1) if m else None


# ================================================================ 模型注册表
def load_registry() -> dict:
    if REGISTRY.exists():
        try:
            return json.loads(REGISTRY.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_registry(reg: dict):
    with _registry_lock:
        REGISTRY.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")


def find_index(exp_name: str) -> Optional[str]:
    """检索索引可能落在 logs/<exp>/ 或 assets/indices/<exp>_added_*.index。"""
    cands = [p for p in sorted((LOGS / exp_name).glob("*.index"), key=lambda p: p.stat().st_mtime)
             if "trained" not in p.name.lower()]
    if not cands:
        cands = [p for p in sorted((RVC_DIR / "assets" / "indices").glob(f"{exp_name}_*added*.index"),
                                   key=lambda p: p.stat().st_mtime)]
    if not cands:
        cands = sorted((RVC_DIR / "assets" / "indices").glob(f"*{exp_name}*added*.index"),
                       key=lambda p: p.stat().st_mtime)
    return cands[-1].relative_to(RVC_DIR).as_posix() if cands else None


def list_models() -> list[dict]:
    reg = load_registry()
    out = {}
    for name, meta in reg.items():
        p = RVC_DIR / meta.get("path", "")
        if p.exists():
            idx = meta.get("index")
            if not (idx and (RVC_DIR / idx).exists()):
                idx = find_index(name)
                if idx:
                    meta["index"] = idx
            out[name] = {
                "name": name,
                "path": meta["path"],
                "index": idx if idx and (RVC_DIR / idx).exists() else None,
                "size_mb": round(p.stat().st_size / 1e6, 1),
                "note": meta.get("note", ""),
                "created": meta.get("created", ""),
                "sr": meta.get("sr", ""),
                "default": bool(meta.get("default")),
            }
    if WEIGHTS.exists():
        for p in sorted(WEIGHTS.glob("*.pth")):
            if p.stem in out:
                continue
            out[p.stem] = {"name": p.stem,
                           "path": p.relative_to(RVC_DIR).as_posix(),
                           "index": find_index(p.stem),
                           "size_mb": round(p.stat().st_size / 1e6, 1),
                           "note": "未登记（自动发现）", "created": "", "sr": "",
                           "default": False}
    return sorted(out.values(), key=lambda m: m["name"])


def default_model() -> Optional[dict]:
    """优先 models.json 里标了 default 的模型，否则用最近创建的。"""
    models = list_models()
    if not models:
        return None
    for m in models:
        if m.get("default"):
            return m
    dated = [m for m in models if m.get("created")]
    if dated:
        return max(dated, key=lambda m: m["created"])
    return models[0]


# ================================================================ 进程工具
def run(cmd, cwd=None, env=None, timeout=3600):
    e = os.environ.copy()
    e.update(CHILD_ENV)
    e["PATH"] = str(BIN_DIR) + os.pathsep + e.get("PATH", "")
    if env:
        e.update(env)
    p = subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None, env=e,
                       timeout=timeout, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise RuntimeError((p.stderr or p.stdout or "命令失败")[-1800:])
    return p


def run_stream(cmd, cwd, env, on_line, job_id, log_file: Path, timeout=None):
    """边跑边读输出，支持取消。timeout=None 表示不限时（训练默认不限时）。"""
    e = os.environ.copy()
    e.update(CHILD_ENV)
    e["PATH"] = str(BIN_DIR) + os.pathsep + e.get("PATH", "")
    e.update(env)
    with open(log_file, "w", encoding="utf-8", errors="replace") as lf:
        p = subprocess.Popen([str(c) for c in cmd], cwd=str(cwd), env=e,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace", bufsize=1)
        t0 = time.time()
        try:
            for line in p.stdout:
                lf.write(line)
                lf.flush()
                if cancelled(job_id):
                    p.terminate()
                    raise RuntimeError("已取消")
                if on_line:
                    on_line(line)
                if timeout and time.time() - t0 > timeout:
                    p.terminate()
                    raise RuntimeError(f"超过设定时限 {timeout/3600:.1f} 小时")
        finally:
            p.wait()
    if p.returncode != 0:
        tail = log_file.read_text(encoding="utf-8", errors="replace")[-1500:]
        raise RuntimeError(tail or "命令失败")


# ================================================================ 取音频
def to_wav(src: Path, dst: Path):
    run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", src,
         "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le", dst])


def fetch_bilibili(bvid: str, dst: Path) -> str:
    import urllib.request

    def api(url: str) -> dict:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Referer": "https://www.bilibili.com"})
        with urllib.request.urlopen(req, timeout=25) as r:
            return json.loads(r.read().decode("utf-8"))

    info = api(f"https://api.bilibili.com/x/web-interface/view?bvid={bvid}")
    if info.get("code") != 0:
        raise RuntimeError(f"取视频信息失败: {info.get('message')}")
    data = info["data"]
    cid = data.get("cid") or (data.get("pages") or [{}])[0].get("cid")
    title = data.get("title", bvid)

    play = api(f"https://api.bilibili.com/x/player/playurl?bvid={bvid}&cid={cid}&fnval=4048&fourk=1")
    if play.get("code") != 0:
        raise RuntimeError(f"取播放地址失败: {play.get('message')}")
    audios = ((play.get("data") or {}).get("dash") or {}).get("audio") or []
    if not audios:
        raise RuntimeError("该视频没有可用音频流（可能是图文/付费内容）")
    best = max(audios, key=lambda a: a.get("bandwidth", 0))
    url = best.get("baseUrl") or best.get("base_url")
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Referer": f"https://www.bilibili.com/video/{bvid}"})
    with urllib.request.urlopen(req, timeout=300) as r, open(dst, "wb") as f:
        shutil.copyfileobj(r, f)
    return title


def fetch_generic(url: str, dst_stem: Path):
    try:
        import yt_dlp
        opts = {"format": "bestaudio/best", "outtmpl": str(dst_stem) + ".%(ext)s",
                "quiet": True, "no_warnings": True, "noplaylist": True,
                "nocheckcertificate": True}
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
        found = sorted(dst_stem.parent.glob(dst_stem.name + ".*"))
        if not found:
            raise RuntimeError("下载完成但没找到音频文件")
        return ((info or {}).get("title") or "audio"), found[0]
    except Exception as err:
        if re.search(r"\.(mp3|wav|m4a|flac|aac|ogg)(\?|$)", url, re.I):
            import urllib.request
            out = dst_stem.with_suffix(Path(url.split("?")[0]).suffix or ".bin")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=300) as r, open(out, "wb") as f:
                shutil.copyfileobj(r, f)
            return url.split("/")[-1], out
        raise RuntimeError(f"无法解析该链接：{str(err)[:300]}")


# ================================================================ 音频处理
def separate(job_dir: Path, src_wav: Path, method: str = "roformer") -> Path:
    """method: roformer（BS-RoFormer 模型分离）| inst（官方伴奏相减）"""
    if method == "inst":
        out = job_dir / "extracted"
        inst_path = job_dir / "inst_source.wav"
        report = out / "report.json"
        # subtract_vocal.py：自动对齐（亚样本）+ 自动匹配音量（分块逐频带）+ 版本自检
        run([PY, ROOT / "work" / "subtract_vocal.py",
             "--mix", src_wav, "--inst", inst_path, "--outdir", out,
             "--report", report], cwd=ROOT, timeout=3600)
        if not (out / "vocals.wav").exists():
            raise RuntimeError("伴奏相减失败（检查两份文件是否为同一版本）")
        return out

    out = job_dir / "roformer"
    run([PY, ROOT / "tools" / "sep_roformer.py", "--input", src_wav, "--outdir", out],
        cwd=ROOT, env={"TORCH_HOME": str(ROOT / "tools" / "torch_cache")}, timeout=7200)
    d = out
    if not (d / "vocals.wav").exists():
        raise RuntimeError("人声分离失败")
    return d


def convert_rvc(job_dir: Path, vocal: Path, pitch: int, model: dict) -> Path:
    dst = job_dir / "vocal_rvc.wav"
    args = [PY, "-m", "infer.cli", "--model", model["path"], "--input", vocal,
            "--output", dst, "--pitch", str(pitch), "--f0-method", "rmvpe",
            "--index-rate", "0.75" if model.get("index") else "0",
            "--protect", "0.33", "--rms-mix-rate", "0.75", "--format", "wav"]
    if model.get("index"):
        args += ["--index", model["index"]]
    run(args, cwd=RVC_DIR, env={"PYTHONPATH": str(RVC_DIR), "RVC_CUDA_GRAPH": "0",
                                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    return dst


def mix_and_master(job_dir: Path, vocal: Path,
                   vocal_ref: Optional[Path], inst_path: Optional[Path]) -> Path:
    """把换声后的人声与伴奏混合并母带。

    - vocal_ref: 分离出的原始人声，用它的音量包络给换声人声做电平匹配（最自然）
    - inst_path: 伴奏；None 表示只做母带
    """
    import numpy as np
    import soundfile as sf
    from scipy.signal import butter, sosfilt

    conv, sr = sf.read(vocal, dtype="float32")
    if conv.ndim > 1:
        conv = conv.mean(1)
    if conv.size == 0:
        raise RuntimeError("换声结果为空")

    raw = job_dir / "mix_raw.wav"
    if inst_path is None or not Path(inst_path).exists():
        sf.write(raw, conv, sr, subtype="FLOAT")
    else:
        inst, sr2 = sf.read(inst_path, dtype="float32", always_2d=True)
        if sr != sr2:
            import librosa
            conv = librosa.resample(conv, orig_sr=sr, target_sr=sr2).astype("float32")
            sr = sr2

        sos = butter(2, 55 / (sr / 2), btype="highpass", output="sos")
        conv = sosfilt(sos, conv).astype("float32")

        if vocal_ref is not None and Path(vocal_ref).exists():
            orig, sr_o = sf.read(vocal_ref, dtype="float32", always_2d=True)
            if sr_o != sr:
                import librosa
                orig = librosa.resample(orig.T, orig_sr=sr_o, target_sr=sr).T
            n = min(len(inst), len(orig))
            inst, orig_m = inst[:n], orig[:n].mean(1)
            conv = np.pad(conv, (0, max(0, n - len(conv))))[:n]

            def smooth(x, w):
                return np.convolve(x, np.ones(w) / w, mode="same") if w > 1 else x

            def rms(x, w):
                return np.sqrt(np.maximum(smooth(x ** 2, w), 1e-12))

            win = int(0.05 * sr)
            r_o, r_c = rms(orig_m, win), rms(conv, win)
            gain = np.clip(smooth(r_o / np.maximum(r_c, 1e-5), 33), 0.2, 5.0)
            positive = r_o[r_o > 1e-4]
            floor = max(1e-4, 0.02 * float(np.percentile(positive, 50))) if positive.size else 1e-4
            quiet = r_o < floor
            gain[quiet] = np.minimum(gain[quiet], 1.0)
            conv = conv * gain
        else:
            # 没有原始人声作参考（例如“源是纯人声 + 官方伴奏”）：按整体 RMS 定一个 -1.5 dB 的声部比例
            n = min(len(inst), len(conv))
            inst, conv = inst[:n], conv[:n]
            r_i = float(np.sqrt(np.mean(inst.astype(np.float64) ** 2)) + 1e-12)
            r_c = float(np.sqrt(np.mean(conv.astype(np.float64) ** 2)) + 1e-12)
            conv = conv * float(np.clip(0.84 * r_i / r_c, 0.05, 30.0))

        sf.write(raw, (inst + conv[:, None]).astype("float32"), sr, subtype="FLOAT")

    final = job_dir / "final.wav"
    run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", raw,
         "-af", "alimiter=limit=0.92:attack=5:release=80:level=0,loudnorm=I=-11:TP=-0.8:LRA=9",
         "-ar", "44100", "-c:a", "pcm_s16le", final])
    return final


# ================================================================ 阶段：源音频 / 官方伴奏
def run_source(sid: str, upload: Optional[Path], raw_name: str, url: Optional[str]):
    try:
        d = SESSIONS / sid
        bili = bool(url and is_bilibili(url))
        bvid = bilibili_bvid(url or "")
        origin = "bilibili" if bili else ("upload" if upload else "url")

        if upload is not None:
            new_raw = d / ("source_new_raw" + (upload.suffix or ".wav"))
            _rm([new_raw])
            shutil.move(str(upload), new_raw)
            title = Path(raw_name).stem
            progress(sid, 25, "正在读取上传的音频…")
        else:
            progress(sid, 8, "正在获取视频信息…" if bili else "正在解析链接…")
            if bili:
                if not bvid:
                    raise RuntimeError("没找到 BV 号")
                new_raw = d / "source_new_raw.m4a"
                _rm([new_raw])
                progress(sid, 15, "正在从 B 站下载最高码率音频…")
                title = fetch_bilibili(bvid, new_raw)
            else:
                title, tmp = fetch_generic(url or "", d / "source_new_raw")
                new_raw = Path(tmp)
            if not new_raw.exists():
                raise RuntimeError("下载完成但没找到音频文件")

        progress(sid, 55, "正在提取 44.1 kHz 音频…")
        new_wav = d / "source_new.wav"
        _rm([new_wav])
        to_wav(new_raw, new_wav)
        dur = audio_duration(new_wav)
        if not dur:
            raise RuntimeError("音频解析失败（没有可用的音轨）")

        ext = new_raw.suffix.lower() or ".bin"
        final_raw = d / ("source_raw" + ext)
        display_raw = Path(raw_name).name if upload is not None else \
            (safe_filename(title) + ext)

        progress(sid, 90, "正在整理…")
        with _sessions_lock:
            s = get_session(sid)
            # 换歌：官方伴奏和后面所有产物全部作废
            wipe_inst_files(s)
            wipe_sep_files(s)
            wipe_conv_files(s)
            wipe_mix_files(s)
            _rm(sorted(d.glob("source_raw*")) + [d / "source.wav"] + sorted(d.glob("source_upload*")))
            if final_raw.exists():
                final_raw.unlink(missing_ok=True)
            os.replace(new_raw, final_raw)
            os.replace(new_wav, d / "source.wav")
            s["source"] = {"wav": "source.wav", "raw": final_raw.name, "raw_name": display_raw,
                           "dur": dur, "origin": origin,
                           "origin_url": (url or "")[:500]}
            s["title"] = (title or Path(display_raw).stem or "未命名").strip()[:120]
            for k in ("inst", "sep", "conv", "mix"):
                s.pop(k, None)
            s["job"] = {"key": "source", "status": "done", "progress": 100,
                        "message": f"已获取音频 · {fmt_dur(dur)}", "finished": time.time()}
            s["updated"] = time.time()
            _write_session(s)
    except Exception as e:
        fail_stage(sid, e)


def run_inst(sid: str, upload: Optional[Path], raw_name: str, url: Optional[str]):
    try:
        d = SESSIONS / sid
        bili = bool(url and is_bilibili(url))
        bvid = bilibili_bvid(url or "")
        origin = "bilibili" if bili else ("upload" if upload else "url")

        if upload is not None:
            new_raw = d / ("inst_new_raw" + (upload.suffix or ".wav"))
            _rm([new_raw])
            shutil.move(str(upload), new_raw)
            title = Path(raw_name).stem
            progress(sid, 25, "正在读取伴奏…")
        else:
            progress(sid, 10, "正在从 B 站下载伴奏…" if bili else "正在解析伴奏链接…")
            if bili:
                if not bvid:
                    raise RuntimeError("没找到 BV 号")
                new_raw = d / "inst_new_raw.m4a"
                _rm([new_raw])
                title = fetch_bilibili(bvid, new_raw)
            else:
                title, tmp = fetch_generic(url or "", d / "inst_new_raw")
                new_raw = Path(tmp)
            if not new_raw.exists():
                raise RuntimeError("下载完成但没找到音频文件")

        progress(sid, 55, "正在转成 44.1 kHz WAV…")
        new_wav = d / "inst_new.wav"
        _rm([new_wav])
        to_wav(new_raw, new_wav)
        dur = audio_duration(new_wav)
        if not dur:
            raise RuntimeError("伴奏解析失败（没有可用的音轨）")

        ext = new_raw.suffix.lower() or ".bin"
        final_raw = d / ("inst_raw" + ext)
        display_raw = Path(raw_name).name if upload is not None else (safe_filename(title) + ext)

        progress(sid, 90, "正在整理…")
        with _sessions_lock:
            s = get_session(sid)
            wipe_sep_files(s)
            wipe_conv_files(s)
            wipe_mix_files(s)
            _rm(sorted(d.glob("inst_raw*")) + [d / "inst_source.wav"] + sorted(d.glob("inst_upload*")))
            if final_raw.exists():
                final_raw.unlink(missing_ok=True)
            os.replace(new_raw, final_raw)
            os.replace(new_wav, d / "inst_source.wav")
            s["inst"] = {"wav": "inst_source.wav", "raw": final_raw.name, "raw_name": display_raw,
                         "dur": dur, "origin": origin, "origin_url": (url or "")[:500]}
            for k in ("sep", "conv", "mix"):
                s.pop(k, None)
            s["options"] = {**(s.get("options") or {}),
                            "separator": "inst", "skip_sep": False}
            s["job"] = {"key": "inst", "status": "done", "progress": 100,
                        "message": f"已就绪 · {fmt_dur(dur)}", "finished": time.time()}
            s["updated"] = time.time()
            _write_session(s)
    except Exception as e:
        fail_stage(sid, e)


# ================================================================ 阶段：分离
def run_separate(sid: str, method: str, skip: bool):
    try:
        s = get_session(sid)
        d = SESSIONS / sid
        if not s.get("source"):
            raise RuntimeError("请先获取源音频")
        if not skip:
            if method == "inst" and not s.get("inst"):
                raise RuntimeError("还没有提供官方伴奏，不能使用“官方伴奏相减”")
            if method not in ("roformer", "inst"):
                method = "roformer"
        # 重跑会替换旧产物：先把后续状态作废，失败时界面不会指向已经删掉的文件
        patch_session(sid, drop=("sep", "conv", "mix"))
        with _sessions_lock:
            wipe_sep_files(s)
            wipe_conv_files(s)
            wipe_mix_files(s)

        if skip:
            progress(sid, 20, "正在按“纯人声”整理…")
            inst = (s.get("inst") or {}).get("wav")
            with _sessions_lock:
                s = get_session(sid)
                s["sep"] = {"method": "none", "skipped": True,
                            "vocal": "source.wav", "inst": inst,
                            "dur": (s.get("source") or {}).get("dur"),
                            "inst_dur": (s.get("inst") or {}).get("dur")}
                for k in ("conv", "mix"):
                    s.pop(k, None)
                s["options"] = {**(s.get("options") or {}), "skip_sep": True}
                s["job"] = {"key": "sep", "status": "done", "progress": 100,
                            "message": "已跳过分离（按纯人声处理）", "finished": time.time()}
                s["updated"] = time.time()
                _write_session(s)
            return

        msg = {"roformer": "正在分离人声 / 伴奏（BS-RoFormer）…",
               "inst": "正在用官方伴奏相减提取人声…"}[method]
        progress(sid, 12, "等待显卡空闲…")
        with _gpu_lock:
            progress(sid, 22, msg)
            stems = separate(d, d / "source.wav", method)

        voc = stems / "vocals.wav"
        ins = stems / "no_vocals.wav"
        if not voc.exists():
            raise RuntimeError("分离失败：没有生成人声轨")

        progress(sid, 90, "正在整理分离结果…")
        diag = None
        if method == "inst":
            rp = stems / "report.json"
            if rp.exists():
                try:
                    diag = json.loads(rp.read_text(encoding="utf-8"))
                except Exception:
                    diag = None
        with _sessions_lock:
            s = get_session(sid)
            s["sep"] = {"method": method, "skipped": False,
                        "vocal": voc.relative_to(d).as_posix(),
                        "inst": ins.relative_to(d).as_posix() if ins.exists() else None,
                        "dur": audio_duration(voc),
                        "inst_dur": audio_duration(ins) if ins.exists() else None}
            if diag:
                s["sep"]["diag"] = diag
            for k in ("conv", "mix"):
                s.pop(k, None)
            s["options"] = {**(s.get("options") or {}),
                            "separator": method, "skip_sep": False}
            s["job"] = {"key": "sep", "status": "done", "progress": 100,
                        "message": f"分离完成 · 人声 {fmt_dur(s['sep']['dur'])}",
                        "finished": time.time()}
            s["updated"] = time.time()
            _write_session(s)
    except Exception as e:
        fail_stage(sid, e)


# ================================================================ 阶段：换音色
def run_convert(sid: str, model_name: str, pitch_raw: str):
    try:
        s = get_session(sid)
        d = SESSIONS / sid
        if not s.get("sep"):
            raise RuntimeError("请先完成分离（或勾选“源是纯人声”跳过分离）")
        try:
            pitch = int(pitch_raw)
        except Exception:
            pitch = 0
        pitch = max(-24, min(24, pitch))

        vocal_in = d / s["sep"]["vocal"]
        if not vocal_in.exists():
            raise RuntimeError("人声轨文件已丢失，请重新分离")

        models = {m["name"]: m for m in list_models()}
        model = models.get(model_name or "") or default_model()
        if not model:
            raise RuntimeError("没有可用模型，请先在「训练模型」页训练一个")
        # 重跑前先作废旧产物，失败时不会留下指向旧文件的“假完成”
        patch_session(sid, drop=("conv", "mix"))
        with _sessions_lock:
            wipe_conv_files(s)
            wipe_mix_files(s)
        progress(sid, 12, "等待显卡空闲…")
        with _gpu_lock:
            progress(sid, 35, f"正在换音色（{model['name']}）…")
            vocal = convert_rvc(d, vocal_in, pitch, model)
        if not vocal.exists():
            raise RuntimeError("换声失败：没有生成输出文件")

        dur = audio_duration(vocal)
        progress(sid, 92, "正在整理…")
        with _sessions_lock:
            s = get_session(sid)
            wipe_mix_files(s)
            s["conv"] = {"model": model["name"], "pitch": pitch,
                         "vocal": vocal.name, "dur": dur}
            s.pop("mix", None)
            s["options"] = {**(s.get("options") or {}),
                            "model": model["name"], "pitch": str(pitch)}
            s["job"] = {"key": "conv", "status": "done", "progress": 100,
                        "message": f"换音色完成 · {model['name']} · {pitch:+d} 半音",
                        "finished": time.time()}
            s["updated"] = time.time()
            _write_session(s)
    except Exception as e:
        fail_stage(sid, e)


# ================================================================ 阶段：混音
def run_mix(sid: str):
    try:
        s = get_session(sid)
        d = SESSIONS / sid
        if not s.get("conv"):
            raise RuntimeError("请先完成换音色")
        vocal = d / s["conv"]["vocal"]
        if not vocal.exists():
            raise RuntimeError("换声文件已丢失，请重新换音色")

        sep = s.get("sep") or {}
        if sep and not sep.get("skipped"):
            vocal_ref = d / sep["vocal"] if sep.get("vocal") else None
            inst = d / sep["inst"] if sep.get("inst") else None
        else:
            vocal_ref = None
            inst = d / sep["inst"] if sep.get("inst") else None

        patch_session(sid, drop=("mix",))
        with _sessions_lock:
            wipe_mix_files(s)

        progress(sid, 25, "正在混合伴奏与换声人声…" if inst else "正在做母带处理…")
        final = mix_and_master(d, vocal, vocal_ref, inst)
        mp3 = d / "final.mp3"
        progress(sid, 80, "正在导出 MP3…")
        run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", final,
             "-c:a", "libmp3lame", "-b:a", "320k", mp3])

        dur = audio_duration(final)
        with _sessions_lock:
            s = get_session(sid)
            s["mix"] = {"wav": final.name, "mp3": mp3.name, "dur": dur,
                        "with_inst": bool(inst)}
            s["job"] = {"key": "mix", "status": "done", "progress": 100,
                        "message": f"成品已生成 · {fmt_dur(dur)}", "finished": time.time()}
            s["updated"] = time.time()
            _write_session(s)
    except Exception as e:
        fail_stage(sid, e)


# ================================================================ 对外状态
def _artifact(s: dict, rel: Optional[str], label: str, *, dur=None, playable=True,
              save_name: Optional[str] = None, note: str = "") -> Optional[dict]:
    if not rel:
        return None
    p = SESSIONS / s["id"] / rel
    if not p.exists() or not p.is_file():
        return None
    ext = p.suffix.lower().lstrip(".")
    if dur is None and playable:
        dur = audio_duration(p)
    try:
        v = int(p.stat().st_mtime)
    except Exception:
        v = 0
    return {"label": label,
            "url": f"/files/{s['id']}/{rel}?v={v}",
            "save_name": save_name or p.name,
            "path": rel,
            "ext": ext,
            "size": p.stat().st_size,
            "dur": dur,
            "playable": bool(playable),
            "note": note}


def public_session(s: dict) -> dict:
    sid = s["id"]
    title = s.get("title") or ""
    base_name = safe_filename(title or "翻唱")
    out = {
        "id": sid,
        "title": title,
        "created": s.get("created"),
        "updated": s.get("updated"),
        "dir": str(SESSIONS / sid),
        "dir_rel": f"webapp/sessions/{sid}",
        "job": s.get("job") or {"key": None, "status": "idle", "progress": 0, "message": ""},
        "options": s.get("options") or {},
        "default_model": (default_model() or {}).get("name"),
    }

    src = s.get("source")
    if src:
        arts = []
        a = _artifact(s, src.get("wav"), "源音频", dur=src.get("dur"),
                      note=("已从视频提取 · 44.1 kHz" if src.get("origin") == "bilibili"
                            else "44.1 kHz · 立体声"),
                      save_name=f"{base_name}_源音频.wav")
        if a:
            arts.append(a)
        if src.get("raw"):
            ext = Path(src["raw"]).suffix.lstrip(".").upper()
            a = _artifact(s, src["raw"], f"原始文件（{ext}）", playable=False,
                          save_name=src.get("raw_name") or Path(src["raw"]).name,
                          note="B 站原始音轨" if src.get("origin") == "bilibili" else "上传/下载的原文件")
            if a:
                arts.append(a)
        out["source"] = {"dur": src.get("dur"), "origin": src.get("origin"), "artifacts": arts}

    inst = s.get("inst")
    if inst:
        arts = []
        a = _artifact(s, inst.get("wav"), "官方伴奏", dur=inst.get("dur"),
                      note="44.1 kHz · 混音直接使用这一份",
                      save_name=f"{base_name}_官方伴奏.wav")
        if a:
            arts.append(a)
        if inst.get("raw"):
            ext = Path(inst["raw"]).suffix.lstrip(".").upper()
            a = _artifact(s, inst["raw"], f"原始文件（{ext}）", playable=False,
                          save_name=inst.get("raw_name") or Path(inst["raw"]).name)
            if a:
                arts.append(a)
        out["inst"] = {"dur": inst.get("dur"), "origin": inst.get("origin"), "artifacts": arts}

    sep = s.get("sep")
    if sep:
        arts = []
        skipped = bool(sep.get("skipped"))
        a = _artifact(s, sep.get("vocal"),
                      "人声（源音频，未分离）" if skipped else "人声（原始）",
                      dur=sep.get("dur"),
                      save_name=f"{base_name}_人声.wav",
                      note="跳过了分离" if skipped else "")
        if a:
            arts.append(a)
        a = _artifact(s, sep.get("inst"),
                      "伴奏（你提供的）" if skipped else "伴奏（原始）",
                      dur=sep.get("inst_dur") or sep.get("dur"),
                      save_name=f"{base_name}_伴奏.wav",
                      note="未分离，直接使用官方伴奏" if skipped else "")
        if a:
            arts.append(a)
        diag = sep.get("diag") or None
        out["sep"] = {"method": sep.get("method"), "skipped": skipped, "artifacts": arts}
        if diag:
            g = diag.get("gain_median_db")
            out["sep"]["diag"] = {
                "delay_ms": diag.get("delay_ms"),
                "gain_db": g,
                "gain_span_db": diag.get("gain_span_db"),
                "lf_coherence": diag.get("lf_coherence"),
                "same_version": diag.get("same_version"),
                "text": ("自动对齐 %+.1f ms · 音量补偿 %+.1f dB" % (diag.get("delay_ms") or 0, g or 0))
                        + ("" if diag.get("same_version") is not False
                           else " · ⚠ 低频相干度低，两份可能不是同一版本"),
            }

    conv = s.get("conv")
    if conv:
        arts = []
        pitch = int(conv.get("pitch") or 0)
        note = f"{conv.get('model', '')} · {pitch:+d} 半音"
        a = _artifact(s, conv.get("vocal"), "换声后的人声", dur=conv.get("dur"),
                      save_name=f"{base_name}_换声_{safe_filename(conv.get('model') or 'model')}.wav",
                      note=note)
        if a:
            arts.append(a)
        out["conv"] = {"model": conv.get("model"),
                       "pitch": pitch, "artifacts": arts}

    mix = s.get("mix")
    if mix:
        arts = []
        a = _artifact(s, mix.get("mp3"), "成品", dur=mix.get("dur"),
                      save_name=f"{base_name}_成品.mp3", note="MP3 320k")
        if a:
            arts.append(a)
        a = _artifact(s, mix.get("wav"), "成品（无损）", dur=mix.get("dur"), playable=False,
                      save_name=f"{base_name}_成品.wav", note="WAV 44.1 kHz")
        if a:
            arts.append(a)
        out["mix"] = {"dur": mix.get("dur"), "with_inst": mix.get("with_inst"), "artifacts": arts}

    return out


def summarize(s: dict):
    job = s.get("job") or {}
    if job.get("status") == "running":
        return "running", job.get("message") or "正在处理"
    if job.get("status") == "error":
        return "error", "有步骤失败"
    if s.get("mix"):
        return "done", "已完成"
    if s.get("conv"):
        return "partial", "待混音"
    if s.get("sep"):
        return "partial", "已分离，待换音色"
    if s.get("source"):
        return "partial", "已获取音频"
    return "empty", "新建"


# ================================================================ 训练流程
def set_state(job_id: str, **kw):
    with _jobs_lock:
        _jobs[job_id].update(kw)


def cancelled(job_id: str) -> bool:
    return job_id in _cancel and _cancel[job_id].is_set()


def process_train(job_id: str, name: str, epochs: int, batch: int, files: list[Path],
                  shutdown_after: bool = False):
    exp_dir = LOGS / name
    log = JOBS / job_id / "train.log"
    try:
        with _gpu_lock:
            t0 = time.time()
            exp_dir.mkdir(parents=True, exist_ok=True)
            cfg = RVC_DIR / "configs" / "v2" / "40k.json"
            shutil.copyfile(cfg, exp_dir / "config.json")

            input_dir = JOBS / job_id / "train_in"
            input_dir.mkdir(exist_ok=True)
            for f in files:
                shutil.move(str(f), input_dir / f.name)

            env = {"PYTHONPATH": str(RVC_DIR),
                   "PATH": str(ROOT / "tools" / "bin") + os.pathsep + os.environ["PATH"]}

            set_state(job_id, step="pre", progress=8, message="正在切片（3.7 秒/段）…")
            run([PY, "-m", "train.preprocess", input_dir, 40000, 4, exp_dir, "True", 3.7],
                cwd=RVC_DIR, env=env)
            n_seg = len(list((exp_dir / "0_gt_wavs").glob("*.wav")))
            if n_seg < 5:
                raise RuntimeError(f"素材太短，只切出 {n_seg} 段（建议每次至少 1 分钟连续人声）")
            if cancelled(job_id):
                raise RuntimeError("已取消")

            set_state(job_id, step="f0", progress=18, message=f"提取音高（{n_seg} 段）…")
            run([PY, "-m", "train.dataset.extract_f0", "cuda", 1, 0, 0, exp_dir, "True"],
                cwd=RVC_DIR, env=env)
            if cancelled(job_id):
                raise RuntimeError("已取消")

            set_state(job_id, step="feat", progress=30, message="提取 ContentVec 特征…")
            run([PY, "-m", "train.dataset.extract_hubert_feature", "cuda", 1, 0, 0, exp_dir, "v2", "True"],
                cwd=RVC_DIR, env=env)
            if cancelled(job_id):
                raise RuntimeError("已取消")

            lines = []
            for w in sorted((exp_dir / "0_gt_wavs").glob("*.wav")):
                stem = w.stem
                lines.append(f"logs/{name}/0_gt_wavs/{w.name}|logs/{name}/3_feature768/{stem}.npy|"
                             f"logs/{name}/2a_f0/{w.name}.npy|logs/{name}/2b-f0nsf/{w.name}.npy|0")
            (exp_dir / "filelist.txt").write_text("\n".join(lines), encoding="utf-8")

            steps_per_epoch = max(1, math.ceil(n_seg / batch))
            total = epochs * steps_per_epoch
            state = {"loss": "", "step": 0}

            def on_line(line: str):
                m = re.search(r"\[(\d+),\s*[\d.eE+-]+\]", line)
                if m:
                    state["step"] = int(m.group(1))
                m2 = re.search(r"loss_mel=([\d.]+)", line)
                if m2:
                    state["loss"] = m2.group(1)
                pct = 38 + int(52 * min(1.0, state["step"] / max(1, total)))
                set_state(job_id, progress=pct,
                          message=f"训练中 {state['step']}/{total} 步 · loss_mel {state['loss']}")

            set_state(job_id, step="train", progress=38,
                      message=f"训练 {epochs} 轮（{n_seg} 段，约 {total} 步）…")
            # save_every_epoch 必须能整除总轮数，否则结束时不会落盘 G_2333333.pth
            if epochs >= 10 and epochs % 10 == 0:
                se = 10
            elif epochs % 5 == 0:
                se = 5
            else:
                se = 1
            train_error = None
            try:
                run_stream([PY, "-m", "train.train", "-se", se, "-te", epochs,
                            "-pg", "assets/pretrained_v2/f0G40k.pth",
                            "-pd", "assets/pretrained_v2/f0D40k.pth",
                            "-g", "0", "-bs", batch, "-e", name, "-sr", "40k",
                            "-v", "v2", "-f0", "1", "-l", "1", "-c", "0"],
                           cwd=RVC_DIR, env=env, on_line=on_line, job_id=job_id, log_file=log)
            except Exception as te:
                train_error = te
                # 训练中断（取消/崩溃）时，只要还有已保存的检查点，就尽量把成果救出来
                if not (exp_dir / "G_2333333.pth").exists():
                    raise
                set_state(job_id, message=f"训练中断：{str(te)[:120]}；正在导出最后的检查点…")

            set_state(job_id, step="index", progress=92, message="建立检索索引…")
            run([PY, "-m", "train.train_index", name, "v2", "assets/indices", "8"],
                cwd=RVC_DIR, env=env, timeout=1800)

            set_state(job_id, step="export", progress=96, message="导出模型…")
            g = exp_dir / "G_2333333.pth"
            if not g.exists():
                cands = sorted(exp_dir.glob("G_*.pth"), key=lambda p: p.stat().st_mtime)
                if not cands:
                    raise RuntimeError("训练结束但没找到生成器权重")
                g = cands[-1]
            code = (f"import sys; sys.argv=['x']\n"
                    f"from train.process_ckpt import extract_small_model\n"
                    f"print(extract_small_model({str(g)!r}, {name!r}, '40k', 1, '', 'v2'))")
            run([PY, "-c", code], cwd=RVC_DIR, env=env)

            idx = find_index(name)
            with _registry_lock:
                reg = load_registry()
                was_default = bool((reg.get(name) or {}).get("default"))
                reg[name] = {"path": f"assets/weights/{name}.pth", "index": idx,
                             "sr": "40k", "f0": 1,
                             "note": f"{n_seg} 段 · {epochs} 轮 · {time.strftime('%m-%d %H:%M')}",
                             "created": time.strftime("%Y-%m-%d %H:%M")}
                if was_default:
                    reg[name]["default"] = True
                save_registry(reg)

            pth = WEIGHTS / f"{name}.pth"
            el = (time.time() - t0) / 60
            if train_error is not None:
                msg = f"模型 {name} 已保存（训练被中断：{str(train_error)[:80]}），用时 {el:.0f} 分钟"
            else:
                msg = f"模型 {name} 训练完成，用时 {el:.0f} 分钟"
            shut = False
            if shutdown_after:
                rc = subprocess.run(["shutdown", "/s", "/f", "/t", "120"],
                                    capture_output=True, text=True, timeout=20)
                shut = rc.returncode == 0
                if shut:
                    msg += "；系统将在 120 秒后关机（执行 shutdown /a 可取消）"
                else:
                    msg += "；关机命令失败：" + (rc.stderr or rc.stdout or "").strip()[:200]
            set_state(job_id, status="done", step="done", progress=100, message=msg,
                      shutdown_scheduled=shut, partial=train_error is not None,
                      result={"model": name, "segments": n_seg,
                              "seconds": round(time.time() - t0),
                              "size_mb": round(pth.stat().st_size / 1e6, 1) if pth.exists() else None})
    except Exception as e:
        set_state(job_id, status="error", step="error", message=str(e)[:800])


# ================================================================ 路由：页面 / 模型
@app.get("/", response_class=HTMLResponse)
def index():
    resp = HTMLResponse((BASE / "index.html").read_text(encoding="utf-8"))
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.get("/api/models")
def api_models():
    models = list_models()
    d = default_model()
    return {"models": models, "default": d["name"] if d else None}


@app.post("/api/models/default")
def api_models_default(name: str = Form(...)):
    models = {m["name"]: m for m in list_models()}
    if name not in models:
        raise HTTPException(404, "没有这个模型")
    with _registry_lock:
        reg = load_registry()
        for k in reg:
            reg[k].pop("default", None)
        if name in reg:
            reg[name]["default"] = True
        else:
            m = models[name]
            reg[name] = {"path": m["path"], "index": m.get("index"), "sr": "40k", "f0": 1,
                         "note": "未登记（自动发现）", "created": "", "default": True}
        save_registry(reg)
    return {"ok": True, "default": name}


# ================================================================ 路由：会话
@app.post("/api/sessions")
def create_session():
    sid = uuid.uuid4().hex[:10]
    (SESSIONS / sid).mkdir(parents=True, exist_ok=True)
    s = {"id": sid, "created": time.time(), "updated": time.time(), "title": "",
         "job": {"key": None, "status": "idle", "progress": 0, "message": ""},
         "options": {"separator": "roformer", "skip_sep": False,
                     "model": "", "pitch": "0"}}
    with _sessions_lock:
        _sessions[sid] = s
        _write_session(s)
    return {"id": sid}


@app.get("/api/sessions")
def list_sessions(limit: int = 10):
    limit = max(1, min(int(limit or 10), 50))
    rows = []
    try:
        dirs = sorted([p for p in SESSIONS.iterdir() if p.is_dir()],
                      key=lambda p: p.stat().st_mtime, reverse=True)
    except Exception:
        return rows
    for d in dirs:
        p = d / "session.json"
        if not p.exists():
            continue
        try:
            s = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        state, label = summarize(s)
        rows.append({"id": s.get("id") or d.name,
                     "title": s.get("title") or "未命名任务",
                     "created": s.get("created") or 0,
                     "updated": s.get("updated") or 0,
                     "state": state, "label": label})
        if len(rows) >= limit:
            break
    return rows


@app.get("/api/sessions/{sid}")
def api_session(sid: str):
    s = get_session(sid)
    return public_session(s)


@app.post("/api/sessions/{sid}/rename")
def api_rename(sid: str, name: str = Form(...)):
    title = re.sub(r"[\x00-\x1f]", " ", str(name or "")).strip()
    title = title[:80] if title else "未命名任务"
    with _sessions_lock:
        s = get_session(sid)
        s["title"] = title
        s["updated"] = time.time()
        _write_session(s)
    return {"ok": True, "title": title}


@app.post("/api/sessions/{sid}/reveal")
def api_reveal(sid: str):
    get_session(sid)
    d = SESSIONS / sid
    if not d.exists():
        raise HTTPException(404, "任务目录不存在")
    try:
        if os.name == "nt":
            subprocess.Popen(["explorer", str(d)])
        else:
            subprocess.Popen(["xdg-open", str(d)])
    except Exception as e:
        raise HTTPException(500, f"打开文件夹失败：{str(e)[:200]}")
    return {"ok": True, "dir": str(d)}


@app.delete("/api/sessions/{sid}")
def api_delete_session(sid: str):
    with _sessions_lock:
        s = get_session(sid)
        if (s.get("job") or {}).get("status") == "running":
            raise HTTPException(409, "任务正在处理，先等它完成再删除")
        d = SESSIONS / sid
        try:
            if d.exists():
                shutil.rmtree(d)
        except Exception as e:
            raise HTTPException(500, f"删除失败（可能有文件正在使用）：{str(e)[:200]}")
        _sessions.pop(sid, None)
        _active_sids.discard(sid)
    return {"ok": True}


@app.post("/api/sessions/{sid}/source")
async def api_source(sid: str, url: str = Form(""), file: Optional[UploadFile] = File(None)):
    get_session(sid)
    ensure_idle(sid)
    d = SESSIONS / sid
    if file is not None and file.filename:
        name = Path(file.filename).name
        ext = (Path(name).suffix or ".wav").lower()
        if ext not in AUDIO_EXT:
            ext = ".bin"
        dst = d / ("source_upload" + ext)
        with open(dst, "wb") as f:
            shutil.copyfileobj(file.file, f)
        claim_job(sid, "source", "正在上传音频…")
        start_stage(sid, run_source, (dst, name, None))
        return {"ok": True}
    u = (url or "").strip()
    if not u:
        raise HTTPException(400, "请选择音频文件，或粘贴 B 站 / 直链")
    claim_job(sid, "source", "正在解析链接…")
    start_stage(sid, run_source, (None, "", u))
    return {"ok": True}


@app.post("/api/sessions/{sid}/inst")
async def api_inst(sid: str, url: str = Form(""), file: Optional[UploadFile] = File(None)):
    get_session(sid)
    ensure_idle(sid)
    d = SESSIONS / sid
    if file is not None and file.filename:
        name = Path(file.filename).name
        ext = (Path(name).suffix or ".wav").lower()
        if ext not in AUDIO_EXT:
            ext = ".bin"
        dst = d / ("inst_upload" + ext)
        with open(dst, "wb") as f:
            shutil.copyfileobj(file.file, f)
        claim_job(sid, "inst", "正在上传伴奏…")
        start_stage(sid, run_inst, (dst, name, None))
        return {"ok": True}
    u = (url or "").strip()
    if not u:
        raise HTTPException(400, "请选择伴奏文件，或粘贴 B 站 / 直链")
    claim_job(sid, "inst", "正在解析伴奏链接…")
    start_stage(sid, run_inst, (None, "", u))
    return {"ok": True}


@app.post("/api/sessions/{sid}/inst/clear")
def api_inst_clear(sid: str):
    ensure_idle(sid)
    clear_inst(sid)
    with _sessions_lock:
        s = get_session(sid)
        s["job"] = {"key": "inst", "status": "idle", "progress": 0,
                    "message": "已清除官方伴奏", "finished": time.time()}
        s["updated"] = time.time()
        _write_session(s)
    return {"ok": True}


@app.post("/api/sessions/{sid}/separate")
def api_separate(sid: str, method: str = Form("roformer"), skip: bool = Form(False)):
    get_session(sid)
    ensure_idle(sid)
    if method not in ("roformer", "inst", "none"):
        raise HTTPException(400, "未知的分离器")
    claim_job(sid, "sep", "正在准备分离…")
    start_stage(sid, run_separate, (method if not skip else "none", bool(skip)))
    return {"ok": True}


@app.post("/api/sessions/{sid}/sep/clear")
def api_sep_clear(sid: str):
    """取消“跳过分离”：作废当前分离结果，回到可以重新分离的状态。"""
    ensure_idle(sid)
    with _sessions_lock:
        s = get_session(sid)
        patch_session(sid, drop=("sep", "conv", "mix"))
        wipe_sep_files(s)
        wipe_conv_files(s)
        wipe_mix_files(s)
        s["options"] = {**(s.get("options") or {}), "skip_sep": False}
        s["job"] = {"key": "sep", "status": "idle", "progress": 0,
                    "message": "已取消跳过分离", "finished": time.time()}
        s["updated"] = time.time()
        _write_session(s)
    return {"ok": True}


@app.post("/api/sessions/{sid}/convert")
def api_convert(sid: str, model: str = Form(""), pitch: str = Form("0")):
    get_session(sid)
    ensure_idle(sid)
    if not PITCH_RE.match(str(pitch or "0")):
        raise HTTPException(400, "变调值不合法")
    claim_job(sid, "conv", "正在准备换音色…")
    start_stage(sid, run_convert, (model, str(pitch)))
    return {"ok": True}


@app.post("/api/sessions/{sid}/mix")
def api_mix(sid: str):
    get_session(sid)
    ensure_idle(sid)
    claim_job(sid, "mix", "正在准备混音…")
    start_stage(sid, run_mix, ())
    return {"ok": True}


# ================================================================ 路由：训练
@app.post("/api/train")
async def train(name: str = Form(...), epochs: int = Form(100), batch: int = Form(4),
                shutdown_after: bool = Form(False),
                files: list[UploadFile] = File(...)):
    name = name.strip()
    if not NAME_RE.match(name):
        raise HTTPException(400, "模型名只能用字母/数字/下划线/短横线，2~32 位")
    if epochs < 1 or epochs > 2000:
        raise HTTPException(400, "轮数需在 1~2000 之间")
    if batch not in (1, 2, 4, 8):
        raise HTTPException(400, "batch 只能是 1/2/4/8")

    job_id = uuid.uuid4().hex[:10]
    job_dir = JOBS / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for i, f in enumerate(files):
        if not f.filename:
            continue
        ext = Path(f.filename).suffix.lower() or ".wav"
        dst = job_dir / f"in_{i:02d}{ext}"
        with open(dst, "wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(dst)
    if not saved:
        raise HTTPException(400, "请至少上传一个音频文件")

    with _jobs_lock:
        _jobs[job_id] = {"id": job_id, "kind": "train", "status": "running",
                         "step": "pre", "progress": 0,
                         "message": f"训练 {name}：{len(saved)} 个文件", "title": f"训练 {name}",
                         "created": time.time(), "outputs": {}}
    _cancel[job_id] = threading.Event()
    threading.Thread(target=process_train, daemon=True,
                     args=(job_id, name, epochs, batch, saved, shutdown_after)).start()
    return {"id": job_id}


@app.post("/api/cancel/{job_id}")
def cancel(job_id: str):
    with _jobs_lock:
        if job_id not in _jobs:
            raise HTTPException(404, "任务不存在")
    if job_id in _cancel:
        _cancel[job_id].set()
    return {"ok": True}


@app.post("/api/shutdown/cancel")
def cancel_shutdown():
    """取消已排队的关机（shutdown -a）。"""
    try:
        subprocess.run(["shutdown", "/a"], capture_output=True, text=True, timeout=10)
        return {"ok": True, "message": "已取消关机"}
    except Exception as e:
        return {"ok": False, "message": str(e)[:200]}


@app.get("/api/jobs/{job_id}")
def job_state(job_id: str):
    with _jobs_lock:
        if job_id not in _jobs:
            raise HTTPException(404, "任务不存在")
        job = dict(_jobs[job_id])
    return JSONResponse(job)


@app.get("/api/jobs")
def job_list():
    with _jobs_lock:
        items = sorted(_jobs.values(), key=lambda x: x["created"], reverse=True)[:20]
    return [{k: i.get(k) for k in ("id", "kind", "title", "status", "progress", "message")}
            for i in items]


# ================================================================ 路由：文件
@app.get("/files/{sid}/{name:path}")
def get_file(sid: str, name: str):
    if not SID_RE.match(sid or ""):
        raise HTTPException(404, "文件不存在")
    base = (SESSIONS / sid).resolve()
    p = (base / name).resolve()
    if base not in p.parents or not p.is_file():
        raise HTTPException(404, "文件不存在")
    media = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    if p.suffix.lower() in (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".webm", ".opus"):
        media = media if media.startswith("audio/") else "audio/wav"
    return FileResponse(str(p), media_type=media)


def _tidy_console() -> None:
    """让控制台里的中文不再乱码。

    Windows 默认代码页是 GBK(936)，而本文件是 UTF-8：把控制台代码页切到
    UTF-8(65001)，并把标准输出/错误固定成 UTF-8 + replace，这样中文日志
    在控制台和重定向到文件时都不会变成“缈诲敱…”或直接抛 UnicodeEncodeError。
    """
    if os.name == "nt":
        try:
            import ctypes
            k32 = ctypes.windll.kernel32
            k32.SetConsoleOutputCP(65001)
            k32.SetConsoleCP(65001)
        except Exception:
            pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


if __name__ == "__main__":
    _tidy_console()
    port = int(os.environ.get("PORT", "7860"))
    print(f"\n  浮声 翻唱工作台  http://127.0.0.1:{port}\n")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
