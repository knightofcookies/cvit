import os
from huggingface_hub import snapshot_download

os.mkdir('deepseekocr', exist_ok=True)
os.mkdir('gotocr2', exist_ok=True)

snapshot_download(repo_id="deepseek-ai/DeepSeek-OCR", local_dir='./deepseekocr')
snapshot_download(repo_id="stepfun-ai/GOT-OCR2_0", local_dir='./gotocr2')

with open('./deepseekocr/__init__.py', 'w') as f:
    f.write('# DeepSeek-OCR package\n')

with open('./gotocr2/__init__.py', 'w') as f:
    f.write('# GOT-OCR2_0 package\n')

