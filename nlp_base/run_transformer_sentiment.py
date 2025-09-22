"""
运行Transformer情感分类器的简化脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transformer_sentiment_classifier import main

if __name__ == "__main__":
    print("启动Transformer情感分类器...")
    main()
