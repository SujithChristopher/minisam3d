@echo off
rem Offline start-to-finish NOARK student pipeline (train -> ONNX -> score vs mocap, every held-out subject,
rem then the all-subject model). Safe to re-run: finished steps are skipped, training resumes from last.pt.
cd /d %~dp0..
"E:\CMC\pyprojects\sam-3d-body\.venv\Scripts\python.exe" scripts\run_noark_loso.py %*
