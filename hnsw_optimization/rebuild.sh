#!/bin/bash

echo "========================================"
echo "重新编译 HNSW C++ 扩展"
echo "========================================"

cd /workspace/vectordbindexing/hnsw_optimization

# 清理旧的构建文件
echo "清理旧的构建文件..."
rm -rf build/
rm -f *.so

# 创建 build 目录
mkdir -p build
cd build

# 使用 CMake 编译
echo "开始编译..."
cmake ..
make -j$(nproc)

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✅ 编译成功！"
    echo "========================================"
    echo ""
    echo "复制编译产物..."
    cp *.so ../
    cd ..
    echo ""
    echo "现在可以测试新功能了："
    echo "  jupyter notebook test_hnsw_optimization.ipynb"
else
    echo ""
    echo "========================================"
    echo "❌ 编译失败！请检查错误信息"
    echo "========================================"
    exit 1
fi

