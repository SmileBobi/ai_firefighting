"""
网络问题解决方案
提供多种pip源和安装方法
"""

import subprocess
import sys
import os

def run_command(command):
    """运行命令并返回结果"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def test_pip_sources():
    """测试不同的pip源"""
    sources = [
        "https://pypi.org/simple/",
        "https://pypi.tuna.tsinghua.edu.cn/simple/",
        "https://mirrors.aliyun.com/pypi/simple/",
        "https://pypi.douban.com/simple/",
        "https://mirrors.cloud.tencent.com/pypi/simple/"
    ]
    
    print("🔍 测试pip源连接...")
    print("=" * 50)
    
    working_sources = []
    
    for source in sources:
        print(f"测试源: {source}")
        success, stdout, stderr = run_command(f"pip install -i {source} --dry-run numpy")
        
        if success:
            print(f"✅ {source} - 可用")
            working_sources.append(source)
        else:
            print(f"❌ {source} - 不可用")
            if stderr:
                print(f"   错误: {stderr.strip()}")
    
    return working_sources

def install_with_best_source():
    """使用最佳源安装依赖"""
    print("\n🚀 使用最佳源安装依赖...")
    print("=" * 50)
    
    # 测试源
    working_sources = test_pip_sources()
    
    if not working_sources:
        print("❌ 所有pip源都不可用，请检查网络连接")
        return False
    
    best_source = working_sources[0]
    print(f"\n使用最佳源: {best_source}")
    
    # 基础依赖
    basic_packages = [
        "numpy",
        "requests", 
        "python-dotenv"
    ]
    
    # 可选依赖
    optional_packages = [
        "pyaudio",
        "librosa",
        "soundfile",
        "scipy"
    ]
    
    print("\n安装基础依赖...")
    for package in basic_packages:
        print(f"安装 {package}...")
        success, stdout, stderr = run_command(f"pip install -i {best_source} {package}")
        if success:
            print(f"✅ {package} 安装成功")
        else:
            print(f"❌ {package} 安装失败: {stderr.strip()}")
    
    print("\n安装可选依赖...")
    for package in optional_packages:
        print(f"安装 {package}...")
        success, stdout, stderr = run_command(f"pip install -i {best_source} {package}")
        if success:
            print(f"✅ {package} 安装成功")
        else:
            print(f"⚠️ {package} 安装失败（可选依赖）: {stderr.strip()}")
    
    return True

def create_offline_install_script():
    """创建离线安装脚本"""
    script_content = '''#!/bin/bash
# 离线安装脚本

echo "📦 创建离线安装包..."

# 创建临时目录
mkdir -p offline_packages
cd offline_packages

# 下载包文件
echo "下载基础依赖包..."
pip download numpy requests python-dotenv

echo "下载可选依赖包..."
pip download pyaudio librosa soundfile scipy

echo "✅ 离线包创建完成"
echo "📁 包文件位置: $(pwd)"
echo ""
echo "🚀 在目标机器上安装:"
echo "pip install --find-links . --no-index numpy requests python-dotenv"
'''
    
    with open("offline_install.sh", "w", encoding="utf-8") as f:
        f.write(script_content)
    
    print("📝 创建离线安装脚本: offline_install.sh")
    print("💡 在有网络的环境运行此脚本下载包文件")

def show_alternative_solutions():
    """显示替代解决方案"""
    print("\n🛠️ 替代解决方案")
    print("=" * 50)
    
    print("1. 使用简化版演示程序（推荐）:")
    print("   python simple_voice_demo.py")
    print("   - 无需安装依赖")
    print("   - 完整功能演示")
    print("   - 模拟真实场景")
    
    print("\n2. 使用conda安装:")
    print("   conda install numpy requests python-dotenv")
    print("   - 通常更稳定")
    print("   - 自动解决依赖")
    
    print("\n3. 使用虚拟环境:")
    print("   python -m venv voice_env")
    print("   voice_env\\Scripts\\activate")
    print("   pip install 包名")
    
    print("\n4. 手动下载whl文件:")
    print("   - 访问 https://pypi.org/project/包名/")
    print("   - 下载对应版本的whl文件")
    print("   - pip install 本地文件路径")
    
    print("\n5. 使用Docker:")
    print("   - 创建Dockerfile")
    print("   - 在容器中安装依赖")
    print("   - 避免环境问题")

def main():
    """主函数"""
    print("🔧 网络问题解决方案")
    print("=" * 50)
    
    while True:
        print("\n请选择解决方案:")
        print("1. 测试pip源连接")
        print("2. 使用最佳源安装依赖")
        print("3. 创建离线安装脚本")
        print("4. 显示替代解决方案")
        print("5. 运行简化版演示程序")
        print("0. 退出")
        
        choice = input("请输入选择 (0-5): ").strip()
        
        if choice == "0":
            break
        elif choice == "1":
            test_pip_sources()
        elif choice == "2":
            install_with_best_source()
        elif choice == "3":
            create_offline_install_script()
        elif choice == "4":
            show_alternative_solutions()
        elif choice == "5":
            print("\n🚀 运行简化版演示程序...")
            os.system("python simple_voice_demo.py")
        else:
            print("❌ 无效选择")

if __name__ == "__main__":
    main()



