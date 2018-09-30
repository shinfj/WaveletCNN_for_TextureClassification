@echo off
cd /d %~dp0

rem Usage: test_waveletcnn.bat path/to/image
rem Image file can also be received with drag & drop to .bat file.
rem This program can process several images at the same time. In that case, those images will be processed one by one.

for %%f in (%*) do (
  echo in processing...
  python run_waveletcnn.py -p test -t %%f 2>NUL
)

pause > NUL
