#!/usr/bin/env bash
# =============================================================================
# Install system-level dependencies for deepfake-detector.
#
# Run once before:  pip install -r requirements.txt
#
# Usage:
#   bash scripts/install_system_deps.sh
#
# What this installs:
#   portaudio    — required by sounddevice (audio I/O)
#   libsndfile   — required by soundfile / librosa (audio decoding)
#   ffmpeg       — required by librosa for mp3/aac/ogg decoding
#   libGL        — required by mediapipe / OpenCV (Linux only)
#   libnotify    — provides notify-send for desktop alerts (Linux only)
#   python3-dev  — C headers needed to compile webrtcvad (Linux only)
#   hdf5         — required by h5py for MesoNet Keras conversion (optional)
#
# System audio loopback drivers (not installed here — see README):
#   macOS:   brew install blackhole-2ch
#   Linux:   PulseAudio monitor source is built-in; no extra install needed
#   Windows: WASAPI loopback is built-in; no extra install needed
# =============================================================================

set -euo pipefail

detect_os() {
    if [[ "$OSTYPE" == darwin* ]]; then
        echo "macos"
    elif [[ -f /etc/debian_version ]]; then
        echo "debian"
    elif [[ -f /etc/fedora-release ]] || [[ -f /etc/redhat-release ]]; then
        echo "fedora"
    elif [[ "$OSTYPE" == msys* ]] || [[ "$OSTYPE" == cygwin* ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

OS=$(detect_os)
echo "Detected OS: $OS"
echo ""

# -----------------------------------------------------------------------------
# macOS
# -----------------------------------------------------------------------------
if [[ "$OS" == "macos" ]]; then
    if ! command -v brew &>/dev/null; then
        echo "Homebrew not found. Install it from https://brew.sh then re-run."
        exit 1
    fi

    echo "Installing system deps via Homebrew..."
    brew install portaudio libsndfile ffmpeg

    echo ""
    echo "Optional: install HDF5 for MesoNet Keras conversion"
    echo "  brew install hdf5 && pip install h5py"
    echo ""
    echo "For system audio capture, install BlackHole:"
    echo "  brew install blackhole-2ch"
    echo "Then configure a Multi-Output Device in Audio MIDI Setup."
    echo "See README.md § macOS loopback setup for details."

# -----------------------------------------------------------------------------
# Debian / Ubuntu
# -----------------------------------------------------------------------------
elif [[ "$OS" == "debian" ]]; then
    echo "Installing system deps via apt..."
    sudo apt-get update -qq
    sudo apt-get install -y \
        portaudio19-dev \
        libsndfile1-dev \
        ffmpeg \
        libgl1 \
        libnotify-bin \
        python3-dev

    echo ""
    echo "Optional: install HDF5 for MesoNet Keras conversion"
    echo "  sudo apt install libhdf5-dev && pip install h5py"
    echo ""
    echo "For system audio capture, a PulseAudio monitor source is usually"
    echo "available by default. Run:"
    echo "  python scripts/list_audio_devices.py"
    echo "and look for a device with 'monitor' in its name."

# -----------------------------------------------------------------------------
# Fedora / RHEL
# -----------------------------------------------------------------------------
elif [[ "$OS" == "fedora" ]]; then
    echo "Installing system deps via dnf..."

    # Check if RPM Fusion is enabled (needed for ffmpeg)
    if ! dnf repolist | grep -q rpmfusion-free; then
        echo ""
        echo "NOTE: ffmpeg requires RPM Fusion. To enable it:"
        echo "  sudo dnf install https://mirrors.rpmfusion.org/free/fedora/rpmfusion-free-release-\$(rpm -E %fedora).noarch.rpm"
        echo "Then re-run this script."
        echo ""
        SKIP_FFMPEG=1
    else
        SKIP_FFMPEG=0
    fi

    PKGS=(
        portaudio-devel
        libsndfile-devel
        mesa-libGL
        libnotify
        python3-devel
    )
    if [[ "$SKIP_FFMPEG" == "0" ]]; then
        PKGS+=(ffmpeg)
    fi

    sudo dnf install -y "${PKGS[@]}"

    echo ""
    echo "Optional: install HDF5 for MesoNet Keras conversion"
    echo "  sudo dnf install hdf5-devel && pip install h5py"

# -----------------------------------------------------------------------------
# Windows
# -----------------------------------------------------------------------------
elif [[ "$OS" == "windows" ]]; then
    echo "Windows detected."
    echo ""
    echo "Most dependencies are bundled in the Python wheels:"
    echo "  sounddevice — PortAudio bundled"
    echo "  soundfile   — libsndfile bundled"
    echo ""
    echo "You need to install FFmpeg manually:"
    echo "  Option A (Chocolatey):  choco install ffmpeg"
    echo "  Option B: download from https://ffmpeg.org/download.html"
    echo "            extract and add the bin/ folder to your PATH"
    echo ""
    echo "Verify FFmpeg is on PATH:  ffmpeg -version"

# -----------------------------------------------------------------------------
# Unknown
# -----------------------------------------------------------------------------
else
    echo "Unknown OS. Please install the following manually:"
    echo "  - PortAudio (audio I/O)"
    echo "  - libsndfile (audio decoding)"
    echo "  - FFmpeg (audio decoding)"
    echo "  - libGL / OpenGL (face detection)"
    echo "See README.md § 1 for details."
    exit 1
fi

echo ""
echo "System deps done. Next:"
echo "  pip install -r requirements.txt"
echo "  python main.py"
