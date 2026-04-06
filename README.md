# Real-Time Deepfake Detector

Monitors video calls and audio-only calls for AI-generated or face-swapped content, firing OS-level notifications when a likely deepfake is detected.

Works with any call app (Zoom, FaceTime, Google Meet, Teams, phone-over-internet) by capturing the screen and/or incoming audio rather than hooking into specific applications.

---

## How it works

### Video pipeline

1. **Screen capture** — `mss` grabs frames from the display at configurable intervals (default: 8 frames spread over the first 30 seconds of a call, then one burst per minute thereafter).
2. **Face detection** — MediaPipe BlazeFace locates the remote participant's face in each frame in under 1 ms.
3. **Deepfake classification** — the face crop is passed to a ViT or CNN classifier. Majority vote across the burst determines the result.

### Audio pipeline

1. **Mic VAD gate** — WebRTC VAD (Google's voice activity detector, same one used in Chrome) continuously monitors your microphone. Analysis only runs when *you* are silent — so you're always classifying the other person's voice.
2. **System audio capture** — 3-second clips of incoming audio are buffered during your silence windows.
3. **Deepfake classification** — clips are passed to a wav2vec2 or HuBERT classifier fine-tuned on anti-spoofing data.

### Notifications

When a modality exceeds the confidence threshold, a native OS notification fires (Notification Center on macOS, `libnotify` on Linux, Windows Toast on Windows). A 30-second cooldown prevents spam.

```
┌─────────────────────────────────────────────────────┐
│  Screen / camera frames                             │
│       │                                             │
│  MediaPipe BlazeFace ──► face crop                  │
│       │                                             │
│  CommunityForensics ViT / GenConViT / MesoNet       │
│       │                                             │
│  majority vote ──► OS notification (if fake)        │
├─────────────────────────────────────────────────────┤
│  Microphone stream ──► WebRTC VAD                   │
│                              │                      │
│                    user silent?                     │
│                              │ yes                  │
│  System audio stream ──► 3-sec clip buffer          │
│                              │                      │
│  wav2vec2-xlsr / HuBERT classifier                  │
│                              │                      │
│  threshold check ──► OS notification (if fake)      │
└─────────────────────────────────────────────────────┘
```

---

## System requirements

- **OS**: macOS 12+, Ubuntu 20.04+, or Windows 10+ (64-bit)
- **Python**: 3.10 or newer
- **RAM**: 4 GB minimum; 8 GB recommended (wav2vec2 model loads ~1.2 GB)
- **Disk**: ~3 GB for model cache (HuggingFace `~/.cache`)
- **CPU**: any modern x86-64 or Apple Silicon chip. GPU/MPS optional but not required.

---

## 1. System-level dependencies

These must be installed before `pip install -r requirements.txt`.

### macOS

```bash
brew install portaudio libsndfile ffmpeg
```

`portaudio` is required by `sounddevice` for audio I/O.  
`libsndfile` is required by `librosa`.  
`ffmpeg` is used by `librosa` for audio decoding.

For system audio capture (incoming call audio), install **BlackHole** — a free virtual audio loopback driver:

```bash
brew install blackhole-2ch
```

> Alternatively download the installer from https://github.com/ExistentialAudio/BlackHole.  
> After installing, go to **System Settings → Sound** and create a Multi-Output Device that includes both your speakers and BlackHole 2ch. Set BlackHole as the input device in the detector (see [macOS loopback setup](#macos-loopback-setup) below).

### Ubuntu / Debian

```bash
sudo apt update
sudo apt install \
    portaudio19-dev \
    libsndfile1-dev \
    ffmpeg \
    libgl1 \
    libnotify-bin \
    python3-dev
```

| Package | Required by |
|---|---|
| `portaudio19-dev` | `sounddevice` |
| `libsndfile1-dev` | `librosa` / `soundfile` |
| `ffmpeg` | `librosa` audio decoding |
| `libgl1` | `mediapipe` / OpenCV |
| `libnotify-bin` | `plyer` (provides `notify-send`) |
| `python3-dev` | `webrtcvad` native extension |

For system audio capture (PulseAudio monitor source — usually pre-installed):

```bash
# Verify a monitor source exists:
pactl list sources short | grep monitor
# It will look like: "alsa_output.pci-0000_00_1f.3.analog-stereo.monitor"
# Use its index with --system-audio-device
python scripts/list_audio_devices.py
```

If you are on PipeWire (Ubuntu 22.04+), the PipeWire PulseAudio compatibility layer exposes monitor sources the same way.

### Fedora / RHEL

```bash
sudo dnf install \
    portaudio-devel \
    libsndfile-devel \
    ffmpeg \
    mesa-libGL \
    libnotify \
    python3-devel
```

> `ffmpeg` may require the RPM Fusion repository:  
> `sudo dnf install https://mirrors.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm`

### Windows

- **PortAudio**: bundled inside the `sounddevice` PyPI wheel — no separate install needed.
- **FFmpeg**: download the GPL build from https://ffmpeg.org/download.html and add `ffmpeg.exe` to your `PATH`. Or with Chocolatey: `choco install ffmpeg`.
- **libsndfile**: bundled inside the `soundfile` wheel.
- **libnotify**: not needed — `plyer` uses native Windows Toast notifications.

For system audio capture, `sounddevice` on Windows exposes WASAPI loopback sources automatically. Run `python scripts/list_audio_devices.py` and look for a device with `(loopback)` in its name.

### MesoNet conversion only (optional)

If you want to use the lightweight MesoNet backend, `h5py` is needed to read the Keras weight file. It requires HDF5 libraries:

```bash
# macOS
brew install hdf5
pip install h5py

# Ubuntu / Debian
sudo apt install libhdf5-dev
pip install h5py

# Windows — h5py ships pre-built wheels, no system dep needed
pip install h5py
```

---

## 2. Python dependencies

```bash
pip install -r requirements.txt
```

Key packages and what they do:

| Package | Version | Role |
|---|---|---|
| `torch` | ≥ 2.0 | PyTorch inference engine |
| `transformers` | ≥ 4.35 | Loads CommunityForensics ViT, wav2vec2, HuBERT from HuggingFace |
| `huggingface_hub` | ≥ 0.19 | Model weight downloads (GenConViT, MesoNet conversion) |
| `timm` | ≥ 0.9 | ConvNeXt / Swin Transformer for GenConViT |
| `mediapipe` | ≥ 0.10 | BlazeFace sub-ms face detection |
| `opencv-python` | ≥ 4.8 | Image I/O and colour conversion |
| `sounddevice` | ≥ 0.4.6 | Cross-platform audio capture (wraps PortAudio) |
| `librosa` | ≥ 0.10 | Log-mel spectrogram extraction (LCNN backend) |
| `webrtcvad` | ≥ 2.0.10 | Google WebRTC voice activity detection |
| `mss` | ≥ 9.0 | Cross-platform screen capture |
| `plyer` | ≥ 2.1 | Cross-platform OS notifications |

### GPU / accelerator support

The default install pulls CPU-only PyTorch. For GPU or Apple Silicon acceleration:

```bash
# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Apple Silicon (MPS) — the standard wheel includes MPS
pip install torch torchvision
```

The detector auto-detects MPS → CUDA → CPU in that priority order. Pass `--device cpu` to override.

---

## 3. Model weights

The two default models download automatically on first run via HuggingFace:

| Backend | Model | Size | Auto-download |
|---|---|---|---|
| `community_forensics` (video default) | `buildborderless/CommunityForensics-DeepfakeDet-ViT` | ~85 MB | Yes |
| `genconvit` (video alt) | `Deressa/GenConViT` | ~300 MB | Yes |
| `wav2vec2` (audio default) | `Gustking/wav2vec2-large-xlsr-deepfake-audio-classification` | ~1.2 GB | Yes |
| `hubert` (audio alt) | `abhishtagatya/hubert-base-960h-itw-deepfake` | ~360 MB | Yes |
| `mesonet` (video lightweight) | Converted from `DariusAf/MesoNet` Keras weights | <1 MB | Manual (see below) |

Weights are cached in `~/.cache/huggingface/hub` and reused on subsequent runs.

### MesoNet (lightweight offline option)

```bash
pip install h5py requests tqdm   # conversion deps
python scripts/convert_mesonet_keras.py
# Downloads Meso4_DF.h5 from github.com/DariusAf/MesoNet and converts to .pth
```

### Advanced: fine-tune EfficientNet on FaceForensics++

For higher video accuracy, fine-tune EfficientNet-B0 on FaceForensics++ (request access at https://github.com/ondyari/FaceForensics):

```bash
python scripts/finetune_efficientnet.py --data /path/to/faceforensics
```

### Advanced: train the LCNN audio model on ASVspoof 2019

Register and download ASVspoof 2019 LA from https://datashare.ed.ac.uk/handle/10283/3336, then:

```bash
python scripts/train_audio_lcnn.py --data /path/to/asvspoof2019
```

---

## 4. macOS loopback setup

To capture the incoming call audio (the remote party's voice) on macOS you need a virtual loopback driver. **BlackHole** is the recommended free option.

1. Install BlackHole: `brew install blackhole-2ch`
2. Open **Audio MIDI Setup** (Applications → Utilities).
3. Click **+** → **Create Multi-Output Device**.
4. Check both **BlackHole 2ch** and your normal speakers/headphones.
5. In your call app (Zoom, FaceTime, etc.), set the *output* device to this Multi-Output Device.
6. Find BlackHole's device index:
   ```bash
   python scripts/list_audio_devices.py
   # Look for "BlackHole 2ch" — note its index, e.g. 4
   ```
7. Pass that index when starting the detector:
   ```bash
   python main.py --system-audio-device 4
   ```

Your call audio will still play through your speakers while BlackHole forwards a copy to the detector.

---

## 5. Usage

```bash
# Default: video + audio, best available models
python main.py

# Audio-only call (e.g. phone-over-internet, no video)
python main.py --no-video

# Specify the loopback device for system audio
python main.py --system-audio-device 4

# List all audio devices (to find your loopback index)
python main.py --list-audio-devices

# Lightweight mode — MesoNet video (no internet after first setup) + HuBERT audio
python main.py --video-backend mesonet --audio-backend hubert

# Restrict screen capture to the region showing the remote party's face
# (left, top, width, height in screen pixels)
python main.py --region 960 0 960 1080

# Tune sensitivity — lower threshold = more sensitive, more false positives
python main.py --video-threshold 0.50 --audio-threshold 0.55

# Run for 60 seconds then exit (useful for testing)
python main.py --duration 60

# Print model reference card
python scripts/download_models.py --info
```

### Full CLI reference

```
python main.py [OPTIONS]

Detection control:
  --no-video                 Disable video pipeline
  --no-audio                 Disable audio pipeline

Video options:
  --video-backend BACKEND    community_forensics | genconvit | mesonet
                             (default: community_forensics)
  --sample-count N           Frames per burst (default: 8)
  --sample-window SEC        Spread early burst over this many seconds (default: 30)
  --ongoing-interval SEC     Seconds between re-sampling bursts (default: 60)
  --region L T W H           Screen region to capture in pixels
  --video-threshold P        Fake probability to flag a face (default: 0.60)

Audio options:
  --audio-backend BACKEND    wav2vec2 | hubert | lcnn  (default: wav2vec2)
  --mic-device IDX           Microphone device index (default: OS default)
  --system-audio-device IDX  Loopback/monitor device index
  --audio-threshold P        Fake probability to flag audio (default: 0.60)

General:
  --notification-cooldown S  Min seconds between repeat alerts (default: 30)
  --device DEVICE            cpu | cuda | mps  (default: auto)
  --duration SEC             Exit after this many seconds
  --list-audio-devices       Print audio devices and exit
  --log-level LEVEL          DEBUG | INFO | WARNING | ERROR
```

---

## 6. Model reference

### Video models

#### CommunityForensics ViT — CVPR 2025 `[default]`
- **Paper**: "Community Forensics: Using Thousands of Generators to Train Deepfake Detectors", Park & Owens, CVPR 2025. [arXiv:2411.04125](https://arxiv.org/abs/2411.04125)
- **GitHub**: https://github.com/JeongsooP/Community-Forensics
- **HuggingFace**: `buildborderless/CommunityForensics-DeepfakeDet-ViT`
- **Architecture**: ViT-Small fine-tuned on 2.7 M images from 4,803 distinct generators.
- **Key finding**: Detection accuracy scales with generator diversity in training data, and keeps improving as new generators are added — the most generalisation-focused open model available.
- **License**: MIT. ~658k downloads/month.

#### GenConViT — 2023
- **Paper**: "GenConViT: Generative Convolutional Vision Transformer for Deepfake Video Detection", Deressa et al. [arXiv:2307.07036](https://arxiv.org/abs/2307.07036)
- **GitHub**: https://github.com/erprogs/GenConViT
- **HuggingFace**: `Deressa/GenConViT`
- **Architecture**: ConvNeXt + Swin Transformer backbone with VAE sub-network that learns the latent distribution of real faces. Ensemble of both sub-networks at inference.
- **Results**: 95.8% average accuracy, 99.3% AUC across DFDC, FF++, DeepfakeTIMIT, Celeb-DF v2.
- **License**: GPL-3.0.

#### MesoNet-4 — ICCVW 2018 `[lightweight]`
- **Paper**: "MesoNet: a Compact Facial Video Forgery Detection Network", Afchar et al. [arXiv:1809.00888](https://arxiv.org/abs/1809.00888)
- **GitHub**: https://github.com/DariusAf/MesoNet
- **Architecture**: 4 convolutional layers + 2 FC layers. ~28K parameters. Runs in <2 ms/face on CPU.
- **Weights**: converted from original Keras `.h5` file (real, publicly available in the GitHub repo) via `scripts/convert_mesonet_keras.py`.

#### SOTA references (not yet integrated)
| Method | Venue | Key idea | Weights |
|---|---|---|---|
| SBI | CVPR 2022 Oral | Self-blended training images; strong cross-dataset | Google Drive (see repo) |
| LAA-Net | CVPR 2024 | EfficientNet-B4 + localised artefact attention | Dropbox (see repo) |
| DeepfakeBench | NeurIPS 2023 | 36 detectors, one framework | ImageNet backbone weights |
| UniFD | CVPR 2023 | Frozen CLIP ViT-L/14 + linear head | In repo (`pretrained_weights/`) |
| FatFormer | CVPR 2024 | CLIP + frequency-domain adapter | Google Drive / OneDrive |

### Audio models

#### wav2vec2-large-xlsr — ASVspoof 2019 `[default]`
- **HuggingFace**: `Gustking/wav2vec2-large-xlsr-deepfake-audio-classification`
- **Architecture**: wav2vec2-large-xlsr-53 (300 M params) fine-tuned for binary real/fake classification.
- **Training data**: ASVspoof 2019 Logical Access partition.
- **EER**: 4.01% on ASVspoof 2019 LA eval. **License**: Apache 2.0. ~150k downloads/month.

#### HuBERT-base in-the-wild
- **HuggingFace**: `abhishtagatya/hubert-base-960h-itw-deepfake`
- **Architecture**: HuBERT-Base (94 M params) fine-tuned on self-described "in-the-wild" deepfake audio.
- **Claimed EER**: 1.43% / 98.7% accuracy. Training data not documented — treat these numbers with caution. **License**: Apache 2.0. ~123k downloads/month.

#### LCNN on log-mel spectrograms `[trainable fallback]`
- **Architecture**: LCNN-9 with Max-Feature-Map activations (Lavrentyeva et al., Interspeech 2017). Input: 64-bin log-mel spectrogram.
- **Training**: `scripts/train_audio_lcnn.py` on ASVspoof 2019 LA (register at https://datashare.ed.ac.uk/handle/10283/3336).

#### SOTA references (not yet integrated)
| Method | Venue | EER | Weights |
|---|---|---|---|
| AASIST | ICASSP 2022 | 0.83% (ASVspoof 2019 LA) | In repo (`models/weights/`) |
| AASIST-L | ICASSP 2022 | 0.99% (lightweight, 85K params) | In repo |
| SSL+AASIST | Odyssey 2022 | 0.82% LA / 2.85% DF | Google Drive (link in README) |
| WavLM ensemble | ASVspoof5 2024 | 6.56% (ASVspoof5) | Not public |

AASIST is at https://github.com/clovaai/aasist (MIT, 275 stars).  
SSL+AASIST is at https://github.com/TakHemlata/SSL_Anti-spoofing (MIT, 165 stars).

---

## 7. Accuracy and limitations

### The generalisation problem

Academic benchmarks significantly overestimate real-world performance. **Deepfake-Eval-2024** ([arXiv:2503.02857](https://arxiv.org/abs/2503.02857)) tested SOTA open models on deepfakes collected from social media in 2024 and found:

- ~50% AUC drop for video detectors
- ~48% AUC drop for audio detectors

As a concrete example: the NII Yamagishilab wav2vec2 model achieves EER ~2% on ASVspoof benchmarks, but 30% EER on Deepfake-Eval-2024. The field is actively working on this; the CommunityForensics approach (training on thousands of generators) is currently the most principled response to the problem.

**Treat detection output as a probabilistic early-warning signal, not a verdict.**

### Other limitations

- **No temporal consistency check**: each frame/clip is classified independently. A real person making unusual facial movements could be flagged.
- **Compression artifacts**: heavy video compression (e.g. on poor networks) can increase false-positive rates.
- **Screen-only capture**: if a call app applies heavy post-processing (background blur, beauty filters) before rendering to screen, the captured frame may differ from the raw stream.
- **Audio timing**: the VAD silence gate may miss short clips if the user and remote party are speaking simultaneously.
- **Model staleness**: deepfake technology advances rapidly. These models should be treated as a snapshot; re-evaluation against new generators is recommended periodically.

---

## 8. Repo structure

```
deepfake-detector/
├── main.py                          # CLI entry point
├── config.py                        # All configuration knobs
├── requirements.txt                 # Python dependencies
├── setup.cfg
│
├── detector/
│   ├── orchestrator.py              # Ties video + audio pipelines together
│   ├── notify.py                    # OS notification layer (plyer + native fallbacks)
│   ├── video/
│   │   ├── frame_sampler.py         # Screen capture with early-burst strategy
│   │   ├── face_extractor.py        # MediaPipe BlazeFace face detection
│   │   └── model.py                 # CommunityForensics / GenConViT / MesoNet
│   └── audio/
│       ├── capture.py               # Dual-stream: mic VAD gate + system audio
│       ├── vad.py                   # WebRTC VAD wrapper
│       └── model.py                 # wav2vec2 / HuBERT / LCNN
│
├── scripts/
│   ├── list_audio_devices.py        # Find your loopback device index
│   ├── convert_mesonet_keras.py     # Download + convert MesoNet Keras weights
│   ├── download_models.py           # Info + MesoNet conversion launcher
│   ├── finetune_efficientnet.py     # Fine-tune EfficientNet on FaceForensics++
│   └── train_audio_lcnn.py          # Train LCNN on ASVspoof 2019 LA
│
└── models/                          # .pth weight files go here (gitignored)
```

---

## 9. Licence notes

| Component | Licence |
|---|---|
| This codebase | MIT |
| CommunityForensics ViT weights | MIT |
| GenConViT weights / source | GPL-3.0 |
| wav2vec2 / HuBERT HF models | Apache 2.0 |
| MesoNet original weights | Research code (no explicit licence) |
| AASIST (reference) | MIT |
| SSL+AASIST (reference) | MIT |
| SBI (reference) | Research only — contact authors for commercial use |
| LAA-Net (reference) | SNT Academic Licence (non-commercial) |
| FaceForensics++ dataset | Research only — form request required |
| ASVspoof 2019 dataset | Research only — free registration required |

For commercial use, CommunityForensics ViT (MIT) and the wav2vec2/HuBERT models (Apache 2.0) are the only components with permissive licences. GenConViT's GPL-3.0 licence requires derivative works to also be GPL.
