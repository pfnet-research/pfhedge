@echo off
echo ========================================
echo PFHedge 开发环境设置脚本
echo ========================================

echo.
echo 1. 检查conda环境...
conda --version
if %errorlevel% neq 0 (
    echo 错误: conda未安装或不在PATH中
    pause
    exit /b 1
)

echo.
echo 2. 删除旧的pfhedge环境（如果存在）...
conda env remove -n pfhedge -y

echo.
echo 3. 创建新的pfhedge环境...
conda create -n pfhedge python=3.12 -y

echo.
echo 4. 激活pfhedge环境...
call conda activate pfhedge

echo.
echo 5. 安装PyTorch...
pip install torch torchvision torchaudio

echo.
echo 6. 安装项目依赖...
pip install -e ".[dev]"

echo.
echo 7. 验证安装...
python -c "import torch; print('PyTorch版本:', torch.__version__)"
python -c "import pfhedge; print('PFHedge版本:', pfhedge.__version__)"

echo.
echo ========================================
echo 环境设置完成！
echo ========================================
echo.
echo 使用方法:
echo 1. 激活环境: conda activate pfhedge
echo 2. 运行测试: pytest
echo 3. 格式化代码: black .
echo 4. 运行示例: python examples/example_readme.py
echo.
pause 