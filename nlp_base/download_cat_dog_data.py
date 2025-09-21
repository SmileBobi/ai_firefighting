#!/usr/bin/env python
"""
下载和准备猫狗数据集
"""

import os
import urllib.request
import zipfile
import shutil
from pathlib import Path


def download_cat_dog_dataset():
    """下载猫狗数据集"""
    
    # 数据集URL (使用Kaggle的猫狗数据集)
    # 注意：这里使用一个公开的数据集作为示例
    # 实际使用时，您可能需要从Kaggle或其他来源下载数据
    
    data_dir = Path('./data/cats_and_dogs')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("正在准备猫狗数据集...")
    print("请手动下载猫狗数据集并按照以下结构组织：")
    print()
    print("data/cats_and_dogs/")
    print("├── train/")
    print("│   ├── cats/")
    print("│   │   ├── cat.0.jpg")
    print("│   │   ├── cat.1.jpg")
    print("│   │   └── ...")
    print("│   └── dogs/")
    print("│       ├── dog.0.jpg")
    print("│       ├── dog.1.jpg")
    print("│       └── ...")
    print("└── test/")
    print("    ├── cats/")
    print("    │   ├── cat.0.jpg")
    print("    │   ├── cat.1.jpg")
    print("    │   └── ...")
    print("    └── dogs/")
    print("        ├── dog.0.jpg")
    print("        ├── dog.1.jpg")
    print("        └── ...")
    print()
    print("数据集来源：")
    print("1. Kaggle: https://www.kaggle.com/c/dogs-vs-cats")
    print("2. 或者使用torchvision.datasets.ImageFolder从本地文件夹加载")
    print()
    print("如果您有数据集，请将其放置在 data/cats_and_dogs/ 目录下")
    
    # 创建示例目录结构
    train_cats_dir = data_dir / 'train' / 'cats'
    train_dogs_dir = data_dir / 'train' / 'dogs'
    test_cats_dir = data_dir / 'test' / 'cats'
    test_dogs_dir = data_dir / 'test' / 'dogs'
    
    for dir_path in [train_cats_dir, train_dogs_dir, test_cats_dir, test_dogs_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print("目录结构已创建完成！")
    print(f"请将您的猫狗图片分别放入对应的目录中")


def create_sample_dataset():
    """创建示例数据集（用于测试）"""
    import numpy as np
    from PIL import Image
    
    data_dir = Path('./data/cats_and_dogs')
    
    # 创建示例图片
    def create_sample_image(size=(224, 224), color=(128, 128, 128), text="Sample"):
        img = Image.new('RGB', size, color)
        return img
    
    print("创建示例数据集...")
    
    # 创建训练集示例
    for i in range(10):
        # 猫的示例图片
        cat_img = create_sample_image(color=(255, 200, 200), text=f"Cat {i}")
        cat_img.save(data_dir / 'train' / 'cats' / f'cat_{i}.jpg')
        
        # 狗的示例图片
        dog_img = create_sample_image(color=(200, 200, 255), text=f"Dog {i}")
        dog_img.save(data_dir / 'train' / 'dogs' / f'dog_{i}.jpg')
    
    # 创建测试集示例
    for i in range(5):
        # 猫的示例图片
        cat_img = create_sample_image(color=(255, 200, 200), text=f"Test Cat {i}")
        cat_img.save(data_dir / 'test' / 'cats' / f'test_cat_{i}.jpg')
        
        # 狗的示例图片
        dog_img = create_sample_image(color=(200, 200, 255), text=f"Test Dog {i}")
        dog_img.save(data_dir / 'test' / 'dogs' / f'test_dog_{i}.jpg')
    
    print("示例数据集创建完成！")
    print("注意：这只是示例数据，实际训练效果会很差")
    print("请使用真实的猫狗图片数据集进行训练")


if __name__ == "__main__":
    print("猫狗数据集准备工具")
    print("=" * 40)
    
    choice = input("选择操作：\n1. 显示数据集结构说明\n2. 创建示例数据集（用于测试）\n请输入选择 (1/2): ")
    
    if choice == '1':
        download_cat_dog_dataset()
    elif choice == '2':
        create_sample_dataset()
    else:
        print("无效选择")
