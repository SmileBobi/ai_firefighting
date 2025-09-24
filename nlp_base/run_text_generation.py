"""
运行文本生成程序的脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transformer_text_generation import main

if __name__ == "__main__":
    print("🚀 启动文本生成程序...")
    main()
