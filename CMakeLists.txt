cmake_minimum_required(VERSION 3.10)
project(BENCHMARK LANGUAGES CXX CUDA)
option(WITH_OPENMP "Using OpenMP" OFF)
# Enable C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(PKG_CONFIG_USE_CMAKE_PREFIX_PATH TRUE)

find_package (Eigen3 REQUIRED)
if (WITH_CLANG_TIDY)
  find_program (CLANG_TIDY_BIN NAMES "clang-tidy-8" DOC "Path to clang-tidy executable")
  if(NOT CLANG_TIDY_BIN)
    message(STATUS "clang-tidy not found.")
  else()
    message(STATUS "clang-tidy found: ${CLANG_TIDY_EXE}")
    set(DO_CLANG_TIDY "${CLANG_TIDY_BIN}" "-checks=*,-header-filter=.*,-fuchsia-default-arguments")
  endif()
endif()

find_package (CUDA)

find_package(Boost REQUIRED COMPONENTS system program_options)
find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)

if (WITH_OPENMP)
  find_package(OpenMP)
  if (OPENMP_FOUND)
      message("OpenMP enabled")
      set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
      set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
      set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
  endif()
else()
      message("OpenMP disabled")
endif()



INCLUDE_DIRECTORIES(
${CMAKE_CURRENT_SOURCE_DIR}/include
${Boost_INCLUDE_DIR}
)

link_directories(
${CMAKE_BINARY_DIR}/bin
${Boost_LIBRARY_DIRS}
)

set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -pedantic -Wextra -O3 -mtune=native -march=native")
set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -pedantic -Wextra -O0")

add_executable(bench_eigen src/bench_eigen.cpp)
target_link_libraries(bench_eigen PUBLIC ${Boost_LIBRARIES} Threads::Threads Eigen3::Eigen)
if(CLANG_TIDY_BIN)
  set_target_properties(decode PROPERTIES CXX_CLANG_TIDY "${DO_CLANG_TIDY}")
endif()

if (CUDA_FOUND)
  set_source_files_properties (src/bench_cuda.cu PROPERTIES LANGUAGE CUDA)
  add_executable(bench_cuda src/bench_cuda.cu)
  target_link_libraries(bench_cuda PUBLIC ${Boost_LIBRARIES} Threads::Threads Eigen3::Eigen)

  set_source_files_properties (src/bench_cuda2d.cu PROPERTIES LANGUAGE CUDA)
  add_executable(bench_cuda2d src/bench_cuda2d.cu)
  target_link_libraries(bench_cuda2d PUBLIC ${Boost_LIBRARIES} Threads::Threads Eigen3::Eigen)

  set_source_files_properties (src/bench_cudamatmul.cu PROPERTIES LANGUAGE CUDA)
  add_executable(bench_cudamatmul src/bench_cudamatmul.cu)
  target_link_libraries(bench_cudamatmul PUBLIC ${Boost_LIBRARIES} Threads::Threads Eigen3::Eigen)

  set_source_files_properties (src/comp3d.cu PROPERTIES LANGUAGE CUDA)
  add_library(comp3d src/comp3d.cu)
   target_link_libraries(comp3d PUBLIC ${Boost_LIBRARIES} Threads::Threads Eigen3::Eigen)
endif()

#  set_source_files_properties (src/bench_cuda3d.cu PROPERTIES LANGUAGE CUDA)
  add_executable(bench_cuda3d src/bench_cuda3d.cpp)
   target_link_libraries(bench_cuda3d PUBLIC ${Boost_LIBRARIES} Threads::Threads Eigen3::Eigen comp3d)
