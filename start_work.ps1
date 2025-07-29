# PFHedge 工作环境启动脚本 (PowerShell版本)
Write-Host "========================================" -ForegroundColor Green
Write-Host "PFHedge 工作环境启动脚本" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

Write-Host ""
Write-Host "正在激活 pfhedge 环境..." -ForegroundColor Yellow

# 激活conda环境
conda activate pfhedge

Write-Host ""
Write-Host "环境已激活！当前环境信息：" -ForegroundColor Green
python --version
python -c "import pfhedge; print('PFHedge版本:', pfhedge.__version__)"

Write-Host ""
Write-Host "您现在可以开始工作了！" -ForegroundColor Green
Write-Host "提示：输入 'python' 启动Python解释器" -ForegroundColor Cyan
Write-Host "提示：输入 'jupyter notebook' 启动Jupyter" -ForegroundColor Cyan
Write-Host "提示：输入 'exit' 退出当前环境" -ForegroundColor Cyan

Write-Host "" 