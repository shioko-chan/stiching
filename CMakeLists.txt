cmake_minimum_required(VERSION 3.10)
project(DOMGenerator)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE RELEASE)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(ONNXRUNTIME_ROOT /opt/onnxruntime)
set(ONNXRUNTIME_INCLUDE_DIR  ${ONNXRUNTIME_ROOT}/include)
find_library(ONNXRUNTIME_LIB onnxruntime
  PATHS ${ONNXRUNTIME_ROOT}/lib
  REQUIRED
)
find_package(CGAL REQUIRED)
find_package(GMP REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW REQUIRED fftw3)

add_executable(mapstitch mapstitch.cpp imagematch.hpp orthprojection.cpp lightglue.cpp superpoint.cpp imagefile.hpp exif.cpp registration.cpp triangulation.hpp)
target_include_directories(mapstitch PRIVATE ${OpenCV_INCLUDE_DIRS} ${ONNXRUNTIME_INCLUDE_DIR} ${GMP_INCLUDE_DIR} ${FFTW_INCLUDE_DIRS})
target_link_libraries(mapstitch
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIB}
    CGAL::CGAL
    ${GMP_LIBRARIES}
    ${FFTW_INCLUDE_DIRS}
    ${FFTW_LIBRARIES} 
)

