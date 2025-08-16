### Install
- install cmake,gcc
```shell
sudo apt update
# install gcc,g++,make
sudo apt install -y build-essential
# install cmake
sudo apt install -y cmake
```

- install required libs
```shell
sudo apt install libaio-dev libgoogle-perftools-dev clang-format libboost-all-dev libmkl-full-dev

# GTest
cd /usr/src/googletest
sudo cmake -S . -B build
sudo cmake --build build -j
sudo cmake --install build
```

- Build
```shell
rm -rf build && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
```