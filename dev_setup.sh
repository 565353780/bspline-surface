#if [ "$(uname)" = "Darwin" ]; then
#  brew install bear
#elif [ "$(uname)" = "Linux" ]; then
#  sudo apt install bear -y
#fi

pip install -U ninja open3d

pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu124

./compile.sh
