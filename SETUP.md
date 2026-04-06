# Holos Medyk — Setup Guide

Step-by-step setup instructions for all four platforms: Windows laptop (development), Raspberry Pi 5 (standalone device), Android (mobile app), iOS (mobile app).

**Table of contents**
- [1. Windows Laptop (Developer Workstation)](#1-windows-laptop-developer-workstation)
- [2. Raspberry Pi 5 (Standalone Device)](#2-raspberry-pi-5-standalone-device)
- [3. Android Phone](#3-android-phone)
- [4. iOS / iPhone](#4-ios--iphone)

---

## 1. Windows Laptop (Developer Workstation)

This is the dev machine for iterating on the voice pipeline, the training corpus, the fine-tuning script, and the Flutter app builds.

### 1.1 Hardware requirements
- **GPU:** NVIDIA with at least 6 GB VRAM (tested on RTX A1000 6 GB Laptop). CUDA 12.x or 13.x driver.
- **RAM:** **32 GB** — 16 GB will crash the laptop during model load. This is not a suggestion.
- **Disk:** ~30 GB free for models, binaries, Flutter SDK, Android SDK.
- **OS:** Windows 11.

### 1.2 Install core tools

1. **Miniconda** (Python environment manager):
   Download and install from https://docs.conda.io/en/latest/miniconda.html. Default install options.

2. **Git** (if not already installed):
   https://git-scm.com/download/win

3. **CUDA driver:** verify with PowerShell:
   ```powershell
   nvidia-smi
   ```
   Should show your GPU and the CUDA driver version. If not, install/update the NVIDIA GeForce/RTX driver.

### 1.3 Clone the repo

```powershell
cd C:\Users\<you>\Documents\GitHub
git clone https://github.com/<your-username>/holos-medyk.git
cd holos-medyk
```

### 1.4 Create the Python environment

```powershell
conda create -n holos python=3.11 -y --override-channels -c conda-forge
```

**Important:** use Python **3.11** (not 3.13) and use the **conda-forge** channel. The default Anaconda channel has ToS gates that block new environments, and Python 3.13 breaks `faster-whisper` and other wheel availability.

Then activate and install dependencies:

```powershell
conda activate holos
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

The torch install line is separate because it needs the PyTorch CUDA index URL. The `requirements.txt` install pulls in `faster-whisper`, `silero-vad`, `piper-tts`, `sounddevice`, `soundfile`, `huggingface-hub`, and `numpy`.

Verify CUDA is working:

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '-', torch.cuda.get_device_name(0))"
```

Should print `CUDA: True - NVIDIA RTX A1000 6GB Laptop GPU` (or your GPU).

### 1.5 Install llama.cpp prebuilt binaries

No C++ compiler needed — we use the prebuilt Windows CUDA release.

1. Go to https://github.com/ggml-org/llama.cpp/releases and find the latest stable build.
2. Download **both** of these zip files (match your CUDA version; 12.4 works on CUDA 13.x drivers thanks to forward compatibility):
   - `llama-<version>-bin-win-cuda-12.4-x64.zip`
   - `cudart-llama-bin-win-cuda-12.4-x64.zip`
3. Extract **both** zips into `holos-medyk/tools/llama-cpp/` so that `llama-cli.exe` and all the `*.dll` files (including `cudart64_12.dll`, `cublas64_12.dll`, `ggml-cuda.dll`) sit flat in the same directory.

Verify:

```powershell
.\tools\llama-cpp\llama-cli.exe --version
```

### 1.6 Log in to HuggingFace

You'll need a HuggingFace account and a **read token** (https://huggingface.co/settings/tokens).

```powershell
conda activate holos
hf auth login
```

Paste your token when prompted. Note: the CLI is now `hf`, not the older `huggingface-cli` (which is deprecated and not always on PATH in new versions).

### 1.7 Download the GGUF models

```powershell
hf download bartowski/google_gemma-4-E4B-it-GGUF --include "google_gemma-4-E4B-it-Q3_K_M.gguf" --local-dir models/
hf download bartowski/google_gemma-4-E2B-it-GGUF --include "google_gemma-4-E2B-it-Q4_K_M.gguf" --local-dir models/
```

This pulls the Q3_K_M quantization of E4B (~4.6 GB) and Q4_K_M of E2B (~3.3 GB). Q3_K_M was chosen for E4B specifically because it leaves headroom in the 6 GB VRAM budget for KV cache and compute buffers. Don't use Q4_K_M or higher for E4B on a 6 GB GPU.

### 1.8 Download the Piper Ukrainian voices

The plan mentioned `uk_UA-lada-medium` but this voice does not exist in the repo. The actually-available Ukrainian voices are `uk_UA-ukrainian_tts-medium` (better quality, larger) and `uk_UA-lada-x_low` (smaller, lower quality):

```powershell
python -c "import os; from huggingface_hub import hf_hub_download; [hf_hub_download('rhasspy/piper-voices', f, local_dir='voices') for f in ['uk/uk_UA/ukrainian_tts/medium/uk_UA-ukrainian_tts-medium.onnx', 'uk/uk_UA/ukrainian_tts/medium/uk_UA-ukrainian_tts-medium.onnx.json', 'uk/uk_UA/lada/x_low/uk_UA-lada-x_low.onnx', 'uk/uk_UA/lada/x_low/uk_UA-lada-x_low.onnx.json']]"
```

### 1.9 Run the desktop voice pipeline

**Important: set `PYTHONUTF8=1` before running anything that touches Ukrainian text.** Windows defaults to `cp1252` which silently replaces Cyrillic characters with `?` on the command line and in subprocess pipes.

```powershell
$env:PYTHONUTF8 = "1"
conda activate holos
python scripts/pipeline.py --model e4b
```

This loads Silero VAD, faster-whisper (small), llama.cpp + Gemma 4 E4B on GPU, and Piper TTS. Speak in Ukrainian when it shows "Listening..." — it uses end-of-utterance silence detection. Press Ctrl+C to exit.

Other useful commands:

```powershell
# Quick inference test, no mic, no TTS
python scripts/inference.py --file scripts/prompt_uk.txt

# Test TTS voices only
python scripts/test_tts.py --no-play
# or play as it synthesizes
python scripts/test_tts.py

# Faster model for iteration
python scripts/pipeline.py --model e2b

# Better Ukrainian ASR (bigger Whisper model)
python scripts/pipeline.py --asr-model medium
```

### 1.10 Key flags for llama.cpp on this hardware

| Flag | Why |
|---|---|
| `-ngl 99` | Full GPU offload. Partial offload (`-ngl 10-30`) triggers a GGML split assertion bug on Gemma 4. CPU-only (`-ngl 0`) loads the entire model into system RAM and OOMs even on 32 GB. |
| `-c 2048` | Context window. |
| `-n 1024` | Max output tokens. |
| `--single-turn` | Generate once and exit. Without this, `llama-cli` drops into interactive REPL mode. |
| `-f <file>` | Read prompt from a UTF-8 file. Use this for any Cyrillic input — do not use `-p "..."` with Ukrainian text on Windows. |

### 1.11 Measured performance on RTX A1000 6GB / 32GB RAM

| Model | Quant | File Size | VRAM Used | Prompt Processing | Generation |
|---|---|---|---|---|---|
| Gemma 4 E4B | Q3_K_M | 4.6 GB | ~3.0 GB | ~380 t/s | ~30 t/s |
| Gemma 4 E2B | Q4_K_M | 3.3 GB | ~2.0 GB | ~530 t/s | ~60 t/s |

### 1.12 Common pitfalls

- **Laptop freezes during model load.** You're on 16 GB RAM or you forgot `-ngl 99`. Upgrade to 32 GB or use full GPU offload.
- **`FileNotFoundError: google/gemma-4-4b-it`.** The repo name uses uppercase `E4B`, not lowercase `4b`: it's `google/gemma-4-E4B-it`.
- **Piper prints hundreds of "Missing phoneme" warnings.** You're passing Ukrainian text through a `cp1252` subprocess pipe. Set `PYTHONUTF8=1` and pass input as pre-encoded UTF-8 bytes: `subprocess.run(..., input=text.encode("utf-8"))`.
- **Silero VAD throws "Provided number of samples is 480".** Silero requires exactly 512 samples at 16 kHz, not a duration-derived buffer size. Hard-code `chunk_samples = 512`.
- **`huggingface-cli: command not found`.** Use `hf` instead — the CLI was renamed. The package is still `huggingface-hub`.

---

## 2. Raspberry Pi 5 (Standalone Device)

Fresh Pi setup from unboxing to a working Ukrainian voice assistant. Target: **Pi 5 with 8 GB RAM** (4 GB is not enough for E4B).

### 2.1 Hardware you need
- **Raspberry Pi 5, 8 GB** — the CanaKit Aluminum Pro kit includes the Pi, aluminum case with active cooling, pre-loaded 128 GB microSD, USB-C PD power supply, and micro-HDMI cables.
- **Audio input:** a USB microphone (the CanaKit bundles one) **or** a Bluetooth headset with built-in mic.
- **Audio output:** a USB speaker, a Bluetooth speaker/headset, or an HDMI monitor with speakers. **The Pi 5 has no 3.5 mm audio jack**, unlike the Pi 4.
- **Monitor, keyboard, mouse** for first-time setup (can be disconnected later once SSH is working).
- **Stable Wi-Fi** — not a phone hotspot. The model download is ~3.6 GB and mobile caps will burn through quickly.

### 2.2 First boot and Pi OS setup

1. Insert the microSD card into the Pi.
2. Connect monitor, keyboard, mouse, and USB-C power.
3. Pi OS boots into the first-run wizard. Set:
   - Locale, keyboard, time zone.
   - Username and password. **Write the username down** — you'll use it for SSH. For consistency with the existing project, something like `voice-help-ua` works.
   - Wi-Fi network.
4. Let the wizard complete and reach the desktop.
5. Open a terminal and update everything:
   ```bash
   sudo apt update && sudo apt full-upgrade -y
   sudo reboot
   ```

### 2.3 Enable SSH and move to headless workflow

On the Pi (terminal):

```bash
sudo systemctl enable ssh
sudo systemctl start ssh
hostname -I
```

Write down the IP address that `hostname -I` prints (something like `192.168.1.xxx` or `10.xxx.xxx.xxx`). You'll use this to connect from your laptop.

On your laptop (PowerShell):

```powershell
ssh <username>@<pi-ip-address>
```

Enter the password you set. You can now close the monitor/keyboard and work entirely from your laptop. **Note:** `raspberrypi.local` mDNS resolution is unreliable on many hotspot and captive networks — always use the raw IP.

### 2.4 Install system dependencies

On the Pi (via SSH):

```bash
sudo apt install -y python3-pip python3-venv git libportaudio2 espeak-ng pulseaudio pulseaudio-module-bluetooth
```

- `python3-pip`, `python3-venv`: Python package tools.
- `libportaudio2`: needed by `sounddevice` for audio I/O.
- `espeak-ng`: Piper TTS phonemization backend.
- `pulseaudio`, `pulseaudio-module-bluetooth`: Bluetooth audio support (if you'll use a Bluetooth headset).

### 2.5 Create the Python environment

```bash
python3 -m venv ~/holos-env
source ~/holos-env/bin/activate
```

### 2.6 Install LiteRT-LM

```bash
pip install litert-lm
```

This pulls `litert-lm` and `litert-lm-api` (the latter has native wheels for `linux-aarch64` on Python 3.11–3.14, which matches the Pi). This is the runtime that actually runs Gemma 4 on the Pi with native audio input support. **LiteRT-LM does not support Windows**, which is why the laptop setup uses llama.cpp instead.

Also install the Python audio and TTS stack:

```bash
pip install piper-tts sounddevice soundfile numpy huggingface-hub
```

### 2.7 Log in to HuggingFace

```bash
hf auth login
```

Paste your read token.

### 2.8 Download the Gemma 4 E4B LiteRT-LM model

```bash
hf download litert-community/gemma-4-E4B-it-litert-lm gemma-4-E4B-it.litertlm
```

This is ~3.6 GB. The file will be cached under `~/.cache/huggingface/hub/models--litert-community--gemma-4-E4B-it-litert-lm/snapshots/<hash>/gemma-4-E4B-it.litertlm`.

**If the download is interrupted** (e.g. network drops): an unreleased lock file can block a retry. Clean the cache completely and start over:

```bash
rm -rf ~/.cache/huggingface/
hf download litert-community/gemma-4-E4B-it-litert-lm gemma-4-E4B-it.litertlm
```

Find the exact path on disk:

```bash
find ~/.cache -name "*.litertlm" 2>/dev/null
```

You'll use this path in the `pi_pipeline.py` script.

### 2.9 Download the Piper Ukrainian voice

```bash
python3 -c "
import os
from huggingface_hub import hf_hub_download
for f in [
    'uk/uk_UA/ukrainian_tts/medium/uk_UA-ukrainian_tts-medium.onnx',
    'uk/uk_UA/ukrainian_tts/medium/uk_UA-ukrainian_tts-medium.onnx.json',
]:
    hf_hub_download('rhasspy/piper-voices', f, local_dir=os.path.expanduser('~/voices'))
print('Done')
"
```

### 2.10 Configure Bluetooth audio (optional, if not using USB mic/speaker)

1. Put your Bluetooth headset in pairing mode.
2. On the Pi:
   ```bash
   bluetoothctl
   ```
3. Inside `bluetoothctl`:
   ```
   power on
   agent on
   scan on
   ```
   Wait for your device to appear, note its MAC address (e.g. `88:92:CC:C1:19:2E`), then:
   ```
   pair 88:92:CC:C1:19:2E
   trust 88:92:CC:C1:19:2E
   connect 88:92:CC:C1:19:2E
   scan off
   exit
   ```
4. Start PulseAudio and verify the device is visible:
   ```bash
   pulseaudio --start
   pactl list sinks short
   pactl list sources short
   ```
   You should see `bluez_output.XX_XX_XX_XX_XX_XX.1` (speaker) and `bluez_input.XX:XX:XX:XX:XX:XX` (mic). If not, reconnect with `bluetoothctl connect <MAC>`.

If you are using a **USB microphone** and **USB or HDMI speaker** instead, skip this step entirely. USB devices are plug-and-play and appear automatically in `arecord -l` and `aplay -l`.

### 2.11 Get the pipeline script onto the Pi

From your laptop (PowerShell, **not** inside SSH):

```powershell
scp C:\Users\<you>\Documents\GitHub\holos-medyk\scripts\pi_pipeline.py <username>@<pi-ip>:~/pi_pipeline.py
```

### 2.12 Update the model path in the script

If the model path on your new Pi differs from the default hard-coded in `scripts/pi_pipeline.py`, edit it:

```bash
nano ~/pi_pipeline.py
```

Update `MODEL_PATH` to whatever `find ~/.cache -name "*.litertlm"` printed.

### 2.13 Test the pipeline

```bash
source ~/holos-env/bin/activate
python3 ~/pi_pipeline.py --prompt "Моя дочка кровоточить з руки. Що робити?"
```

This runs text-only for a first sanity check (no mic, no audio input). The first invocation is slow — LiteRT-LM compiles XNNPACK kernels for the prefill, decode, verify, and audio-encoder subgraphs, then caches them to `.xnnpack_cache` files alongside the model. Subsequent runs skip this step and are much faster.

Then try voice mode:

```bash
python3 ~/pi_pipeline.py
```

Speak Ukrainian into the microphone when it shows "Listening..." The default recording duration is 7 seconds; adjust with `--duration 10`.

### 2.14 Known Pi performance issue (as of Day 2)

CPU-only LiteRT-LM inference on the Pi 5 is slow:
- **Text-only prompt:** ~2 minutes
- **Audio input + text response + TTS synthesis:** ~10 minutes end-to-end

This is too slow for a product. The obvious next levers are (a) enabling the LiteRT-LM GPU backend for the VideoCore VII (delegates are already registered at load time per the logs but not actively used for inference), (b) keeping a single model process alive across turns instead of reloading per invocation, (c) disabling visible thinking, and (d) reducing max-tokens. This is the first thing to tackle on any new Pi setup.

### 2.15 Common pitfalls on Pi

- **Hotspot data cap.** Model download is ~3.6 GB. Use a stable Wi-Fi connection, not mobile hotspot.
- **Stuck download with no progress.** Delete the cache (`rm -rf ~/.cache/huggingface/`) and restart.
- **`arecord -l` shows no capture devices.** You're using Bluetooth and haven't started PulseAudio, or the device isn't connected. Run `pactl list sources short` instead — PipeWire/PulseAudio devices don't appear in the raw ALSA list.
- **No 3.5 mm jack on Pi 5.** Use USB audio or Bluetooth. The Pi 4 had a jack; the Pi 5 removed it.
- **IP changes after reconnecting to a new network.** Run `hostname -I` on the Pi (via monitor) to find the new IP, then SSH to the new address.
- **`huggingface-cli: command not found`.** Use `hf` instead.
- **First inference is very slow but subsequent calls are faster.** XNNPACK kernel compilation caches to disk on first run. This is expected; don't interrupt it.

---

## 3. Android Phone

Fresh phone setup for installing and running the Flutter app.

### 3.1 Device requirements
- **At least 6 GB RAM** for E4B. 8 GB is comfortable. 4 GB devices (e.g. Samsung A15) will not run E4B reliably — they might run E2B, but expect crashes.
- **At least 6 GB of free storage** (app install + ~4 GB model download on first launch).
- **Android 9.0 (Pie) or newer.**
- A GPU that supports OpenCL is strongly preferred. Snapdragon (Adreno), Tensor (Mali/Tensor GPU), or newer MediaTek (Mali) chips all qualify.

### 3.2 Enable Developer Options and USB Debugging

1. On the phone, open **Settings → About Phone** (on Samsung, this may be under **Settings → About Phone → Software Information**).
2. Find the **Build Number** entry and tap it **7 times** in quick succession. After the 7th tap you'll see "You are now a developer."
3. Go back to **Settings**. A new **Developer Options** menu now appears (usually near the top or under **System**).
4. Open **Developer Options** and turn on:
   - **USB Debugging**

### 3.3 Connect the phone to your laptop

1. Plug the phone into the laptop via USB.
2. On the phone, if a prompt appears asking to **Allow USB debugging from this computer**, tap **Always allow** and **OK**.
3. On the laptop (PowerShell), verify the connection:
   ```powershell
   C:\flutter\bin\flutter.bat devices
   ```
   You should see your phone listed as a connected device.

### 3.4 Prerequisites on the laptop

You need Flutter and the Android toolchain on the laptop to build and deploy the app. If you've already set up the laptop per Section 1, you may still need these extras:

1. **Install Android Studio** from https://developer.android.com/studio. Accept all default options, which will also install the Android SDK.

2. **Install Flutter SDK:**
   - Download the latest stable Windows zip from https://docs.flutter.dev/install/archive (e.g. `flutter_windows_3.41.6-stable.zip`).
   - Extract to `C:\flutter`.
   - Add `C:\flutter\bin` to your PATH. In PowerShell (as your user, not admin):
     ```powershell
     [Environment]::SetEnvironmentVariable("Path", $env:Path + ";C:\flutter\bin", "User")
     ```
     Restart PowerShell after this.

3. **Install Android command-line tools:**
   - Open Android Studio → **More Actions → SDK Manager → SDK Tools** tab.
   - Check **Android SDK Command-line Tools** and click **Apply**.

4. **Accept Android licenses:**
   ```powershell
   flutter doctor --android-licenses
   ```
   Press `y` to all prompts.

5. **Verify everything:**
   ```powershell
   flutter doctor
   ```
   You want green checks on **Flutter**, **Windows Version**, **Android toolchain**, and **Connected device**. The Visual Studio item is for Windows desktop apps and can be ignored.

### 3.5 Build and install the app

From the repo root:

```powershell
cd C:\Users\<you>\Documents\GitHub\holos-medyk\app
C:\flutter\bin\flutter.bat pub get
C:\flutter\bin\flutter.bat run
```

`flutter run` compiles the debug APK, installs it to the connected phone, and attaches a debug session. First build takes 3–5 minutes. Subsequent builds are faster because Gradle caches.

While the app is running in debug mode, you can press:
- `r` — hot reload (apply code changes without restarting the app)
- `R` — hot restart (rebuild state from scratch)
- `q` — quit

### 3.6 First launch behavior

1. The app opens with the dark theme and shows "Завантаження моделі…" (Loading model) at the bottom.
2. On first launch, the app downloads the Gemma 4 model from HuggingFace (~4 GB for E4B, smaller for E2B). This only happens once — the model is cached locally after that.
3. Ensure the phone is on **Wi-Fi**, not cellular, for the first launch.
4. Once downloaded and loaded, the status flips to "Натисніть кнопку та говоріть" (Press the button and speak).

### 3.7 Sideloading the APK without USB debugging (alternative)

If you want to install the app on a phone without a laptop connection, build the APK once and share the file:

```powershell
C:\flutter\bin\flutter.bat build apk --release
```

The APK lands at `app\build\app\outputs\flutter-apk\app-release.apk`. Email it, upload to Google Drive, or copy via USB MTP. On the receiving phone:

1. Open **Settings → Apps → Special App Access → Install unknown apps** (the menu name varies by Android version). Allow your file manager or email app to install unknown apps.
2. Open the APK file in that app. Tap **Install**.

### 3.8 Switching between E4B and E2B

The default model URL is hard-coded in `app/lib/gemma_service.dart`. For weaker devices (4 GB RAM), change:

```dart
const String _modelUrl =
    'https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm/resolve/main/gemma-4-E4B-it.litertlm';
```

to:

```dart
const String _modelUrl =
    'https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm';
```

Then hot-restart (`R`) or rebuild (`flutter run`).

### 3.9 Common pitfalls on Android

- **"Build Number" is missing.** On Samsung, it's nested under **About Phone → Software Information → Build Number**.
- **App crashes or phone reboots on launch.** Your device doesn't have enough RAM for E4B. Switch to E2B or use a flagship device.
- **Lost connection to device during `flutter run`.** The model load spikes memory pressure and the debug bridge can drop. The app itself may still be running fine on the phone — unplug and open it directly on the phone to confirm.
- **"FlutterGemma not initialized."** `main()` in `lib/main.dart` must `await FlutterGemma.initialize()` before `runApp()`. This is the Day 1 bug that we already fixed; if you see it in a new project, check that `main()` is async and awaits the call.
- **Layout overflow ("A RenderFlex overflowed by X pixels").** Wrap long-text children of `Row` widgets in `Expanded` or `Flexible`.
- **Import errors from `flutter_gemma`.** Use `package:flutter_gemma/flutter_gemma.dart` — not the IDE-suggested `package:flutter_gemma/core/api/flutter_gemma.dart`. The top-level file re-exports everything.

---

## 4. iOS / iPhone

iOS builds require a **Mac**. There is no supported way to build an iOS app from Windows or Linux — Apple's code signing, provisioning, and the iOS Simulator all require macOS and Xcode. The options below assume you either have access to a Mac or are willing to use a cloud Mac service for the duration of the build.

### 4.1 Requirements
- **Mac running macOS 14 or newer**, with ~20 GB free disk.
- **Xcode 15 or newer**, installed from the Mac App Store (it's a 10+ GB download — do this first and let it run while you set up the other pieces).
- **iPhone** with at least 6 GB RAM (iPhone 12 Pro / 13 / 14 / 15 / 16 / Pro variants). iOS 17 or newer.
- **Apple ID** signed in to Xcode. A **free** Apple Developer account is enough for 7-day side-loaded builds; a **paid** Apple Developer Program membership ($99/year) is required for TestFlight and App Store distribution.

### 4.2 Install Xcode and command-line tools

On the Mac:

1. Install Xcode from the Mac App Store.
2. Open Xcode once after install to accept the license and let it finish downloading iOS components.
3. Install the command-line tools:
   ```bash
   xcode-select --install
   ```
4. Verify:
   ```bash
   xcodebuild -version
   ```

### 4.3 Install CocoaPods

CocoaPods is the iOS dependency manager Flutter uses. Install via:

```bash
sudo gem install cocoapods
pod --version
```

If `gem install` is slow or fails, use Homebrew instead: `brew install cocoapods`.

### 4.4 Install Flutter SDK on the Mac

1. Download Flutter for macOS from https://docs.flutter.dev/install/archive (pick the matching architecture — Apple Silicon for M1/M2/M3, or Intel). Example:
   ```bash
   cd ~/development
   unzip ~/Downloads/flutter_macos_arm64_3.41.6-stable.zip
   ```
2. Add Flutter to PATH. Edit `~/.zshrc` (or `~/.bash_profile` on older macOS):
   ```bash
   export PATH="$PATH:$HOME/development/flutter/bin"
   ```
3. Reload:
   ```bash
   source ~/.zshrc
   ```
4. Verify:
   ```bash
   flutter --version
   flutter doctor
   ```
   You should see green checks on Flutter, Xcode, and CocoaPods.

### 4.5 Clone the repo on the Mac

```bash
cd ~/dev
git clone https://github.com/<your-username>/holos-medyk.git
cd holos-medyk/app
```

### 4.6 Resolve iOS dependencies

```bash
flutter pub get
cd ios
pod install
cd ..
```

`pod install` downloads the LiteRT-LM iOS runtime and all other native iOS dependencies referenced by `flutter_gemma` and friends. This is the step that doesn't exist on the Android side.

### 4.7 Open the project in Xcode and set the signing team

1. Open the Xcode workspace:
   ```bash
   open ios/Runner.xcworkspace
   ```
   **Do not** open `Runner.xcodeproj` directly — always use the `.xcworkspace` for CocoaPods projects.
2. In Xcode, select the **Runner** target in the left sidebar.
3. Go to the **Signing & Capabilities** tab.
4. Under **Team**, select your Apple ID. (If you don't see any teams, add your Apple ID in **Xcode → Settings → Accounts**.)
5. Change the **Bundle Identifier** to something unique — Apple requires a unique identifier per Apple ID. For example, `com.<yourname>.holosmedyk`. Update this in both the Runner target and any other targets that inherit from it (such as tests).

### 4.8 Run on the iOS Simulator (no physical device needed)

```bash
open -a Simulator
flutter run
```

Flutter auto-detects the booted simulator and installs the app. This lets you verify the UI and non-GPU code paths, but on-device LiteRT-LM inference in the simulator is unstable and slow — the simulator does not emulate the real Metal GPU well enough for LLM inference. Treat the simulator as a UI validation target only, not an inference target.

### 4.9 Run on a physical iPhone

1. Plug the iPhone into the Mac via a Lightning or USB-C cable.
2. On the iPhone, tap **Trust** when prompted. You may also need to enter your passcode.
3. In Xcode, with the iPhone selected in the run target dropdown at the top of the window, click the **Run** (▶) button. First build takes several minutes.
4. On the iPhone, the first time you launch a side-loaded app, iOS blocks it with "Untrusted Developer." Go to **Settings → General → VPN & Device Management → Developer App → Trust "<your Apple ID>"**. Then reopen the app.

With a **free** Apple ID, the provisioning profile expires after 7 days — after that, you'll need to rebuild and redeploy from Xcode. With a **paid** developer account, profiles last a year and you can also distribute through TestFlight.

### 4.10 First launch behavior

The iOS app downloads the Gemma 4 `.litertlm` model on first launch, just like Android. Make sure the iPhone is on Wi-Fi for this. The model lands in the app's documents directory and persists across launches.

### 4.11 Building for TestFlight (paid Apple Developer account required)

With a paid developer account:

```bash
flutter build ipa --release
```

This produces an `.ipa` in `build/ios/ipa/`. Upload it either through Xcode's **Organizer** window (**Window → Organizer → Archives → Distribute App**) or via `xcrun altool` / Transporter. After upload, the build appears in App Store Connect under **TestFlight** and you can invite testers by email or share a public TestFlight link.

### 4.12 Cloud Mac alternatives (if you don't have a Mac)

If you don't have physical access to a Mac, these are the usable options:

- **MacinCloud** (https://www.macincloud.com/) — rent a Mac per hour or month. Comes with Xcode pre-installed. Monthly plans around $30.
- **MacStadium** — more expensive but higher performance, used by CI systems.
- **GitHub Actions macOS runners** — free tier available for open source projects; works for `flutter build ipa` in CI but you still need a physical device or simulator for interactive testing.
- **Codemagic** — Flutter-aware CI with iOS build and TestFlight upload support.

For this project, a MacinCloud monthly subscription around the hackathon submission date is probably the cheapest path to a working TestFlight build if no Mac is available locally.

### 4.13 Common pitfalls on iOS

- **`xcode-select` errors after updating macOS.** Run `sudo xcode-select --reset`.
- **`pod install` fails on M1/M2/M3 Mac with "ffi" or "nokogiri" errors.** Install Ruby via Homebrew first (`brew install ruby`), then reinstall CocoaPods.
- **"No profiles for 'com.example.app' were found".** You forgot to set the Team and/or change the Bundle Identifier in Xcode. Every Apple ID needs a unique bundle id.
- **App launches but LiteRT-LM fails to load the model.** The iOS simulator is not a reliable inference target. Test on a physical device.
- **"Untrusted Developer" on iPhone.** Go to Settings → General → VPN & Device Management and trust your Apple ID.
- **Free provisioning profile expires after 7 days.** Expected behavior for free Apple IDs. Rebuild from Xcode to refresh.
- **Flutter plugin errors specific to iOS.** After every `flutter pub get`, re-run `cd ios && pod install`. Don't skip it.

---

## Quick Reference: Which Runtime on Which Platform

| Platform | Inference Runtime | Model Format | Why |
|---|---|---|---|
| Windows laptop | llama.cpp (prebuilt CUDA binary) | GGUF (Q3_K_M for E4B) | LiteRT-LM has no Windows build; llama.cpp is the closest equivalent |
| Raspberry Pi 5 | LiteRT-LM (Python API) | `.litertlm` | Official runtime with native audio input support |
| Android | LiteRT-LM via `flutter_gemma` plugin | `.litertlm` | Plugin wraps LiteRT-LM for Flutter, supports both GPU and NPU backends |
| iOS | LiteRT-LM via `flutter_gemma` plugin | `.litertlm` | Same plugin, uses Metal GPU acceleration on-device |

All four platforms run the **same** Gemma 4 E4B weights. The only thing that differs is the runtime wrapper and quantization format.
