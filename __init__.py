import os
from pathlib import Path

# cwd = os.getcwd()
cwd = Path.cwd()
print(cwd / "DDPM", type(cwd))
