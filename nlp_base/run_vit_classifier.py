"""
运行ViT图像分类程序的脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vit_image_classifier import main

if __name__ == "__main__":
    print("🚀 启动ViT图像分类程序...")
    main()
