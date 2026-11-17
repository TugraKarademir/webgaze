@echo off
cd /d "%~dp0"

rem Baska bilgisayardan kopyalanan .venv calismaz; calismiyorsa silip yeniden kur
if not exist .venv\Scripts\python.exe goto :kur
.venv\Scripts\python.exe -c "import mediapipe, cv2" >nul 2>&1 && goto :calistir
echo Mevcut Python ortami bu bilgisayarda calismiyor, yeniden kuruluyor...
rmdir /s /q .venv

:kur
echo Python ortami kuruluyor, ilk seferde birkac dakika surer...
rem mediapipe 0.10.14 Python 3.9-3.12 destekler
set PYVER=
for %%v in (3.12 3.11 3.10) do if not defined PYVER py -%%v -c "" >nul 2>&1 && set PYVER=%%v
if not defined PYVER (
    echo Python 3.10, 3.11 veya 3.12 bulunamadi. python.org adresinden Python 3.12 kurun.
    goto :hata
)
py -%PYVER% -m venv .venv || goto :hata
.venv\Scripts\python.exe -m pip install -r requirements.txt || goto :hata

:calistir
.venv\Scripts\python.exe webcam_eye_tracker.py --preview --log gaze.csv %*
if errorlevel 1 goto :hata
exit /b 0

:hata
echo.
echo Bir hata olustu. Yukaridaki mesaji kopyalayip gonderin.
pause
