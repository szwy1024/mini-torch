"""保留原运行命令；框架实现位于 ../mini_torch/，训练代码在 train_text.py。"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.train_text import main

if __name__ == "__main__":
    main()
