#!/usr/bin/env python3
"""
Print all available audio devices so you can identify the
correct loopback/monitor source for system audio capture.

Usage
-----
  python scripts/list_audio_devices.py
"""

import sys


def main() -> None:
    try:
        import sounddevice as sd
    except ImportError:
        print("sounddevice not installed. Run: pip install sounddevice")
        sys.exit(1)

    devices = sd.query_devices()
    hostapis = sd.query_hostapis()

    print(f"\n{'IDX':>4}  {'TYPE':<6}  {'HOSTAPI':<12}  NAME")
    print("-" * 72)
    for i, d in enumerate(devices):
        kind = []
        if d["max_input_channels"] > 0:
            kind.append("IN")
        if d["max_output_channels"] > 0:
            kind.append("OUT")
        api_name = hostapis[d["hostapi"]]["name"] if d["hostapi"] < len(hostapis) else "?"
        default_marker = ""
        try:
            defaults = sd.default.device
            if i == defaults[0]:
                default_marker += " [default-in]"
            if i == defaults[1]:
                default_marker += " [default-out]"
        except Exception:
            pass
        print(f"  {i:>3}  {'/'.join(kind):<6}  {api_name:<12}  {d['name']}{default_marker}")

    print(
        "\nLoopback / system-audio tips:\n"
        "  macOS  — Install BlackHole (https://github.com/ExistentialAudio/BlackHole)\n"
        "           Look for 'BlackHole 2ch' or 'Soundflower (2ch)'\n"
        "  Linux  — Look for a device with 'monitor' in the name (PulseAudio)\n"
        "           e.g.  'Monitor of Built-in Audio Analog Stereo'\n"
        "  Windows — Look for a device with '(loopback)' in the name (WASAPI)\n"
        "\nPass the index to main.py with --system-audio-device IDX\n"
    )


if __name__ == "__main__":
    main()
