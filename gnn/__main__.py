#!/usr/bin/env python3
"""
GNN模块的主入口点
当使用 python -m gnn.train_model 时，会执行这个文件
"""

import sys
import os

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 检查命令行参数，确定要运行的模块
if len(sys.argv) > 1 and sys.argv[1] == 'train_model':
    # 移除 'train_model' 参数，因为main.py不需要它
    sys.argv.pop(1)
    
    # 导入并运行main模块
    from main import main
    main()
else:
    # 默认运行main模块
    from main import main
    main()