# Dots And Boxes AI

An AI agent trained to play the Dots and Boxes game, using Reinforcement Training technique.

## Prerequisite

-   Python >= 3.12
-   Git

## Running The Game Locally

Download the latest release from [releases](https://github.com/ItzShubhamDev/dots-and-boxes-ai/releases/tag/1.0.0) and have fun beating the AI.

## Train Yourself

```bash
git clone https://github.com/ItzShubhamDev/dots-and-boxes-ai
cd dots-and-boxes-ai
python -m venv env
.\env\Scripts\activate
pip install -r requirements.txt
python train.py
```

## Different Files Usage

| File         | Usage                                    |
|--------------|------------------------------------------|
|env.py        |Environment Declaration for Dots and Boxes|
|train.py      |Trains the model                          |
|torchToOnnx.py| Converts the model to ONNX               |
|run.py        |Runs PyGame with PyTorch model             |
|run_onnx.py   |Runs PyGame with ONNX model                |

## PyInstaller Export Spec Files

```spec
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['run_onnx.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('models/dots_and_boxes_model.onnx', 'models'),
        ('env.py', '.'),
    ],
    hiddenimports=[
        'pygame',
        'numpy',
        'onnxruntime',
        'pettingzoo',
        'gymnasium',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='DotsAndBoxes',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
```