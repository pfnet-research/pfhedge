@echo off
echo ========================================
echo PFHedge 工作环境启动脚本
echo ========================================

echo.
echo 正在激活 pfhedge 环境...
call conda activate pfhedge

echo.
echo 环境已激活！当前环境信息：
python --version
python -c "import pfhedge; print('PFHedge版本:', pfhedge.__version__)"

echo.
echo 您现在可以开始工作了！
echo 提示：输入 'python' 启动Python解释器
echo 提示：输入 'jupyter notebook' 启动Jupyter
echo 提示：输入 'exit' 退出当前环境

echo.
cmd /k 