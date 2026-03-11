# hook-torch.py
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = collect_all("torch")

# Remove problematic modules
excludes = [
    'torch.cuda',
    'torch.distributed',
    "torch._dynamo",
    "torch._inductor",
    "torch._numpy",
    "torch.compiler",
    "torch.nn.attention"
]