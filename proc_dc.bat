@echo off

set bindir=c:\VivoFollow\bin\
set cfg=%1

%bindir%\DistCorr_64.exe -cfg:%cfg%
rem %bindir%\DistCorr_64-Debug.exe -cfg:%bindir%\DistCorr_align.cfg -sleep:10000
