name: Build Windows EXE

on:
  push:
    branches: [ "main", "master" ]
  workflow_dispatch:

jobs:
  build:
    runs-on: windows-latest

    steps:
    - name: Checkout Code
      uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: "3.10" 

    - name: Install Dependencies
      # 这一步依然可以用默认 shell，或者也加上 shell: cmd 保持一致
      shell: cmd
      run: |
        python -m pip install --upgrade pip
        
        REM 安装 CPU 版 PyTorch (减小体积)
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        
        REM 安装其他依赖
        pip install pandas sentence-transformers huggingface_hub pyinstaller
        
        REM 验证
        python -c "import torch; print('Torch version:', torch.__version__)"

    - name: Run PyInstaller (Build EXE)
      # 【关键修改】显式指定使用 cmd，这样 ^ 符号才能被正确识别为换行
      shell: cmd
      run: |
        pyinstaller --clean --onefile --name "DataGovernanceTool" ^
          --hidden-import="sklearn.utils._cython_blas" ^
          --hidden-import="sklearn.neighbors.typedefs" ^
          --hidden-import="sklearn.neighbors.quad_tree" ^
          --hidden-import="sklearn.tree._utils" ^
          --hidden-import="sentence_transformers" ^
          run_final_prod.py

    - name: Upload EXE Artifact
      uses: actions/upload-artifact@v4
      with:
        name: DataGovernanceTool_Windows
        path: dist/DataGovernanceTool.exe
