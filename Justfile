set shell := ['bash', '-uc']
data_path := "dataset/datasetColor"
exposure_time := "11000"
iso := "550"

clean:
  rm -r build

cmake:
  [ -d build ] || cmake -S. -Bbuild

build: cmake
  cmake --build build --parallel 2

save: build
  mkdir -p {{data_path}}/{left,right,color}
  rm {{data_path}}/{left,right,color}/*||true
  ./build/save_images {{data_path}} {{exposure_time}} {{iso}}

run: build
  ./build/oakdvio ./calibration/oakd.yaml

kitti: build
  ./build/oakdvio /kitti-dataset/dataset/sequences/00/ ./src/violib/calibration/kitti00.yaml /kitti-dataset/poses/00.txt
  # ./build/oakdvio /kitti-dataset/dataset/sequences/00/ ./calibration/oakd.yaml /kitti-dataset/poses/00.txt

calibration: build
  ./build/calibration_reader
