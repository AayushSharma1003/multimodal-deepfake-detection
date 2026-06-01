@echo off
REM ============================================================
REM setup.bat — FULLY AUTOMATIC setup for deepfake detection
REM
REM USAGE: Just double-click this file. That's it.
REM
REM This script will:
REM   1. Download and install Miniconda (if not present)
REM   2. Create conda env "dfd" with Python 3.11
REM   3. Install PyTorch 2.3.1 + CUDA 12.1
REM   4. Install all pinned dependencies
REM   5. Create Jupyter kernel
REM   6. Create project folders
REM   7. Verify GPU access
REM
REM ESTIMATED TIME: 15-25 minutes (depends on internet speed)
REM ============================================================

echo.
echo ============================================================
echo   DEEPFAKE DETECTION PROJECT — FULLY AUTOMATIC SETUP
echo ============================================================
echo.
echo   This will take 15-25 minutes. Just let it run.
echo   DO NOT close this window.
echo.

REM --- Step 0: Check if conda exists ---
where conda >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Conda already installed.
    goto :create_env
)

REM --- Conda not found: auto-download and install Miniconda ---
echo [1/7] Conda not found. Downloading Miniconda...
echo       (This is a ~80 MB download)

set MINICONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-py311_24.7.1-0-Windows-x86_64.exe
set MINICONDA_EXE=%TEMP%\Miniconda3-installer.exe

REM Download using PowerShell (available on all Windows 10+ machines)
powershell -Command "& { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%MINICONDA_URL%' -OutFile '%MINICONDA_EXE%' -UseBasicParsing }"

if not exist "%MINICONDA_EXE%" (
    echo.
    echo [!] DOWNLOAD FAILED. Please install Miniconda manually:
    echo     1. Open Chrome
    echo     2. Go to: https://docs.conda.io/en/latest/miniconda.html
    echo     3. Download "Miniconda3 Windows 64-bit"
    echo     4. Install it (CHECK "Add to PATH")
    echo     5. Close this window and double-click setup.bat again
    echo.
    pause
    exit /b 1
)

echo [2/7] Installing Miniconda (silent install, this takes 2-5 min)...
echo       Installing to: %USERPROFILE%\Miniconda3

"%MINICONDA_EXE%" /InstallationType=JustMe /RegisterPython=1 /AddToPath=1 /S /D=%USERPROFILE%\Miniconda3

if %errorlevel% neq 0 (
    echo [!] Miniconda installation failed.
    echo     Try installing manually from: https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

REM Clean up installer
del "%MINICONDA_EXE%" 2>nul

REM Add conda to PATH for this session
set PATH=%USERPROFILE%\Miniconda3;%USERPROFILE%\Miniconda3\Scripts;%USERPROFILE%\Miniconda3\condabin;%PATH%

REM Initialize conda for cmd
call conda init cmd.exe >nul 2>&1

REM Verify conda works now
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [OK] Miniconda installed but PATH needs refresh.
    echo     Please CLOSE this window and double-click setup.bat again.
    echo.
    pause
    exit /b 0
)

echo [OK] Miniconda installed successfully!
echo.

:create_env
REM --- Step 3: Create conda environment ---
echo [3/7] Creating conda environment "dfd" (Python 3.11)...

call conda create -n dfd python=3.11 -y
if %errorlevel% neq 0 (
    echo [*] Env "dfd" may already exist. Continuing...
)

REM --- Step 4: Activate environment ---
echo [4/7] Activating "dfd" environment...
call conda activate dfd

if %errorlevel% neq 0 (
    echo [!] Could not activate env.
    echo     Close this window, open a new cmd, and run:
    echo       conda activate dfd
    echo       Then run setup.bat again.
    pause
    exit /b 1
)

echo [OK] Python version:
python --version
echo.

REM --- Step 5: Install ALL dependencies (pinned versions) ---
echo [5/7] Installing PyTorch 2.3.1 + CUDA 12.1...
echo       (This is the biggest download, ~2 GB)
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

echo.
echo [5/7] Installing remaining packages (pinned versions)...

REM numpy MUST be less than 2 for compatibility
pip install numpy==1.26.4

REM facenet-pytorch for MTCNN face detection (--no-deps avoids conflicts)
pip install facenet-pytorch==2.6.0 --no-deps

REM Pillow is needed by facenet-pytorch but --no-deps skipped it
pip install Pillow==10.4.0

REM OpenCV for video frame extraction
pip install opencv-python==4.10.0.84

REM Data science stack
pip install pandas==2.2.2
pip install scikit-learn==1.5.1
pip install matplotlib==3.9.2

REM Progress bars
pip install tqdm==4.66.5

REM Jupyter + kernel
pip install ipykernel==6.29.5
pip install jupyter==1.0.0

REM HuggingFace for dataset download
pip install huggingface_hub==0.24.6

echo.

REM --- Step 6: Create Jupyter kernel ---
echo [6/7] Creating Jupyter kernel "dfd"...
python -m ipykernel install --user --name dfd --display-name "dfd (Python 3.11)"
echo.

REM --- Step 7: Create folder structure ---
echo [7/7] Creating project folders...
set PROJECT_DIR=%USERPROFILE%\Desktop\deepfake_detection
set OUTPUT_DIR=%USERPROFILE%\Desktop\LAV-DF-preprocessed

if not exist "%PROJECT_DIR%\DataPrep" mkdir "%PROJECT_DIR%\DataPrep"
if not exist "%OUTPUT_DIR%\video_tensors" mkdir "%OUTPUT_DIR%\video_tensors"
if not exist "%OUTPUT_DIR%\audio_tensors" mkdir "%OUTPUT_DIR%\audio_tensors"

echo   [OK] %PROJECT_DIR%\DataPrep
echo   [OK] %OUTPUT_DIR%\video_tensors
echo   [OK] %OUTPUT_DIR%\audio_tensors
echo.

REM --- Final verification ---
echo ============================================================
echo   VERIFICATION
echo ============================================================
echo.

python -c "import torch; print(f'  PyTorch:       {torch.__version__}'); print(f'  CUDA:          {torch.cuda.is_available()}'); print(f'  GPU:           {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else '  GPU:           NOT DETECTED (will use CPU)')"
python -c "import torchaudio; print(f'  torchaudio:    {torchaudio.__version__}')"
python -c "import torchvision; print(f'  torchvision:   {torchvision.__version__}')"
python -c "import numpy; print(f'  numpy:         {numpy.__version__}')"
python -c "import cv2; print(f'  opencv:        {cv2.__version__}')"
python -c "import pandas; print(f'  pandas:        {pandas.__version__}')"
python -c "from facenet_pytorch import MTCNN; print('  MTCNN:         OK')"

echo.
echo ============================================================
echo   SETUP COMPLETE!
echo ============================================================
echo.
echo   Everything installed. Now run these in VS Code terminal:
echo.
echo     cd %PROJECT_DIR%\DataPrep
echo     conda activate dfd
echo     python download_dataset.py
echo     python explore_dataset.py
echo     python sample_dataset.py
echo     python preprocess_audio.py
echo     python preprocess_video.py
echo     python build_metadata.py
echo     python dataset.py
echo.
echo ============================================================
echo.
pause
