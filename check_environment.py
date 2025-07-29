#!/usr/bin/env python3
"""
PFHedge 环境检查脚本
用于验证开发环境是否正确配置
"""

import sys
import subprocess
import importlib
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    print("🐍 检查Python版本...")
    version = sys.version_info
    print(f"   当前版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("   ✅ Python版本符合要求 (>=3.8)")
        return True
    else:
        print("   ❌ Python版本不符合要求")
        return False

def check_conda_environment():
    """检查conda环境"""
    print("\n📦 检查conda环境...")
    try:
        result = subprocess.run(['conda', 'info', '--envs'], 
                              capture_output=True, text=True, check=True)
        output = result.stdout
        
        if 'pfhedge' in output and '*' in output:
            print("   ✅ 当前在pfhedge环境中")
            return True
        elif 'pfhedge' in output:
            print("   ⚠️  pfhedge环境存在但未激活")
            return False
        else:
            print("   ❌ pfhedge环境不存在")
            return False
    except Exception as e:
        print(f"   ❌ 无法检查conda环境: {e}")
        return False

def check_core_packages():
    """检查核心包"""
    print("\n📚 检查核心包...")
    packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'pfhedge': 'PFHedge'
    }
    
    all_good = True
    for package, name in packages.items():
        try:
            module = importlib.import_module(package)
            if hasattr(module, '__version__'):
                print(f"   ✅ {name}: {module.__version__}")
            else:
                print(f"   ✅ {name}: 已安装")
        except ImportError:
            print(f"   ❌ {name}: 未安装")
            all_good = False
    
    return all_good

def check_dev_packages():
    """检查开发工具"""
    print("\n🛠️  检查开发工具...")
    dev_packages = {
        'pytest': 'pytest',
        'black': 'black',
        'isort': 'isort',
        'flake8': 'flake8',
        'mypy': 'mypy'
    }
    
    all_good = True
    for package, name in dev_packages.items():
        try:
            importlib.import_module(package)
            print(f"   ✅ {name}: 已安装")
        except ImportError:
            print(f"   ❌ {name}: 未安装")
            all_good = False
    
    return all_good

def check_project_structure():
    """检查项目结构"""
    print("\n📁 检查项目结构...")
    required_files = [
        'pfhedge/__init__.py',
        'pyproject.toml',
        'examples/',
        'tests/'
    ]
    
    all_good = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path}")
            all_good = False
    
    return all_good

def run_basic_test():
    """运行基本测试"""
    print("\n🧪 运行基本测试...")
    try:
        # 测试基本导入
        import torch
        import pfhedge
        from pfhedge.instruments import BrownianStock, EuropeanOption
        from pfhedge.nn import Hedger, MultiLayerPerceptron
        
        print("   ✅ 基本导入测试通过")
        
        # 测试基本功能
        stock = BrownianStock()
        derivative = EuropeanOption(stock)
        model = MultiLayerPerceptron()
        hedger = Hedger(model, inputs=["log_moneyness", "expiry_time"])
        
        print("   ✅ 基本功能测试通过")
        return True
        
    except Exception as e:
        print(f"   ❌ 基本测试失败: {e}")
        return False

def main():
    """主函数"""
    print("=" * 50)
    print("🔍 PFHedge 环境检查")
    print("=" * 50)
    
    checks = [
        ("Python版本", check_python_version),
        ("Conda环境", check_conda_environment),
        ("核心包", check_core_packages),
        ("开发工具", check_dev_packages),
        ("项目结构", check_project_structure),
        ("基本测试", run_basic_test)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"   ❌ {name}检查失败: {e}")
            results.append((name, False))
    
    # 总结
    print("\n" + "=" * 50)
    print("📊 检查总结")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name}: {status}")
    
    print(f"\n总体结果: {passed}/{total} 项检查通过")
    
    if passed == total:
        print("🎉 环境配置正确！可以开始开发了。")
        return 0
    else:
        print("⚠️  环境配置有问题，请参考 '环境管理指南.md' 进行修复。")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 