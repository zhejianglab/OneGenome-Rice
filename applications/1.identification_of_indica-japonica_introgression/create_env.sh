# ---------------------------------------------------------
# 1. 创建并激活环境
# ---------------------------------------------------------
# 建议使用 Python 3.11，兼容性最好
conda create -n env_introgression_analysis python=3.11 -y
conda activate env_introgression_analysis

# ---------------------------------------------------------
# 2. 升级 pip (推荐，避免一些兼容性问题)
# ---------------------------------------------------------
pip install --upgrade pip

# ---------------------------------------------------------
# 3. 安装 PyTorch (GPU 版本, CUDA 12.8)
# ---------------------------------------------------------
# 关键点：必须使用 --index-url 指定官方源
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128

# ---------------------------------------------------------
# 4. 安装其他所有依赖
# ---------------------------------------------------------
pip install numpy==2.2.6 scipy==1.17.1 pandas==2.2.3 scikit-learn==1.8.0
pip install transformers==4.57.6 sentencepiece==0.2.1 joblib==1.5.3 accelerate==1.13.0

echo "✅ 所有包已通过 pip 安装完成！"