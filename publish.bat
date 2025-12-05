@echo off
echo ==========================================
echo NEMUS - Pushing Existing Repository
echo ==========================================

echo Adding remote origin...
git remote add origin https://github.com/ycharankrishna/NEMUS.git

echo Renaming branch to main...
git branch -M main

echo Pushing to GitHub...
git push -u origin main

echo ==========================================
echo DONE!
echo ==========================================
pause
