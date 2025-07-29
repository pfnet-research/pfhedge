#!/usr/bin/env python3
"""
PFHedge 环境自动修复脚本
自动检测并修复常见的环境问题
"""

import sys
import subprocess
import os
from pathlib import Path

def run_command(command, description):
    """运行命令并显示结果"""
    print(f"\n🔧 {description}...")
    print(f"   执行: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print("   ✅ 成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ 失败: {e}")
        print(f"   错误输出: {e.stderr}")
        return False

def check_and_fix_conda():
    """检查并修复conda环境"""
    print("\n📦 检查conda环境...")
    
    # 检查conda是否可用
    try:
        subprocess.run(['conda', '--version'], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("   ❌ conda未安装或不在PATH中")
        print("   请先安装Anaconda或Miniconda")
        return False
    
    # 检查pfhedge环境
    try:
        result = subprocess.run(['conda', 'env', 'list'], 
                              capture_output=True, text=True, check=True)
        if 'pfhedge' in result.stdout:
            print("   ✅ pfhedge环境存在")
            
            # 检查是否激活
            if os.environ.get('CONDA_DEFAULT_ENV') == 'pfhedge':
                print("   ✅ pfhedge环境已激活")
                return True
            else:
                print("   ⚠️  pfhedge环境未激活，正在激活...")
                return run_command("conda activate pfhedge", "激活pfhedge环境")
        else:
            print("   ❌ pfhedge环境不存在，正在创建...")
            return run_command("conda create -n pfhedge python=3.12 -y", "创建pfhedge环境")
    except Exception as e:
        print(f"   ❌ 检查环境失败: {e}")
        return False

def install_core_dependencies():
    """安装核心依赖"""
    print("\n📚 安装核心依赖...")
    
    # 安装PyTorch
    if not run_command("pip install torch torchvision torchaudio", "安装PyTorch"):
        return False
    
    # 安装项目依赖
    if not run_command("pip install -e .", "安装PFHedge项目"):
        return False
    
    return True

def install_dev_dependencies():
    """安装开发依赖"""
    print("\n🛠️  安装开发依赖...")
    
    # 安装开发工具
    dev_packages = [
        "pytest>=6.2.5",
        "black==21.9b0", 
        "isort==5.9.3",
        "flake8>=5.0.0",
        "mypy>=1.11.1"
    ]
    
    for package in dev_packages:
        if not run_command(f"pip install {package}", f"安装 {package}"):
            print(f"   ⚠️  跳过 {package}")
    
    return True

def verify_installation():
    """验证安装"""
    print("\n✅ 验证安装...")
    
    try:
        # 测试导入
        import torch
        import pfhedge
        print(f"   ✅ PyTorch {torch.__version__}")
        print(f"   ✅ PFHedge {pfhedge.__version__}")
        
        # 测试基本功能
        from pfhedge.instruments import BrownianStock, EuropeanOption
        from pfhedge.nn import Hedger, MultiLayerPerceptron
        
        stock = BrownianStock()
        derivative = EuropeanOption(stock)
        model = MultiLayerPerceptron()
        hedger = Hedger(model, inputs=["log_moneyness", "expiry_time"])
        
        print("   ✅ 基本功能测试通过")
        return True
        
    except Exception as e:
        print(f"   ❌ 验证失败: {e}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("🔧 PFHedge 环境自动修复")
    print("=" * 60)
    
    # 检查当前目录
    if not Path("pyproject.toml").exists():
        print("❌ 请在项目根目录运行此脚本")
        return 1
    
    # 修复步骤
    steps = [
        ("检查conda环境", check_and_fix_conda),
        ("安装核心依赖", install_core_dependencies),
        ("安装开发依赖", install_dev_dependencies),
        ("验证安装", verify_installation)
    ]
    
    for step_name, step_func in steps:
        print(f"\n{'='*20} {step_name} {'='*20}")
        if not step_func():
            print(f"\n❌ {step_name}失败，请手动检查")
            return 1
    
    print("\n" + "=" * 60)
    print("🎉 环境修复完成！")
    print("=" * 60)
    print("\n下一步:")
    print("1. 运行测试: pytest")
    print("2. 运行示例: python examples/example_readme.py")
    print("3. 检查环境: python check_environment.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 