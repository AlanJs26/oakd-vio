clean:
  rm -r build

cmake:
  [ -d build ] || cmake -S. -Bbuild

build: cmake
  cmake --build build --parallel 2

run: build
  ./build/oakdvio ./calibration/oakd.yaml

kitti: build
  ./build/oakdvio /kitti-dataset/dataset/sequences/00/ ./src/violib/calibration/kitti00.yaml /kitti-dataset/poses/00.txt
  # ./build/oakdvio /kitti-dataset/dataset/sequences/00/ ./calibration/oakd.yaml /kitti-dataset/poses/00.txt

calibration: build
  ./build/calibration_reader
