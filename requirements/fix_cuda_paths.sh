NVIDIA_PACKAGE_DIR="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia"

for dir in $NVIDIA_PACKAGE_DIR/*; do
    if [ -d "$dir/lib" ]; then
        export LD_LIBRARY_PATH="$dir/lib:$LD_LIBRARY_PATH"
    fi
done