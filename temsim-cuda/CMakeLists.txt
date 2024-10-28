cmake_minimum_required(VERSION 3.10)

project(temsim_gpu LANGUAGES CXX CUDA)

# Set the build type to Release for optimization
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Set the C++ and CUDA standards
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Option to specify FFTW directory
set(FFTW_DIR "" CACHE PATH "Path to FFTW installation")

# Find FFTW headers
find_path(FFTW_INCLUDE_DIR fftw3.h
    HINTS ${FFTW_DIR}/include ${FFTW_DIR}
    PATH_SUFFIXES include
    DOC "FFTW include directory"
)

# Find FFTW libraries
find_library(FFTW_LIBRARY fftw3f
    HINTS ${FFTW_DIR}/lib ${FFTW_DIR}/lib64 ${FFTW_DIR}
    PATH_SUFFIXES lib lib64
    DOC "FFTW single-precision library"
)

find_library(FFTW_THREADS_LIBRARY fftw3f_threads
    HINTS ${FFTW_DIR}/lib ${FFTW_DIR}/lib64 ${FFTW_DIR}
    PATH_SUFFIXES lib lib64
    DOC "FFTW single-precision threads library"
)

if(FFTW_INCLUDE_DIR AND FFTW_LIBRARY AND FFTW_THREADS_LIBRARY)
    set(FFTW_LIBRARIES ${FFTW_THREADS_LIBRARY} ${FFTW_LIBRARY})
else()
    message(FATAL_ERROR "Could not find FFTW libraries (fftw3f and fftw3f_threads).")
endif()

# Find Threads package
find_package(Threads REQUIRED)

# Define the sources for the TEMSIM library
set(TEMSIM_LIB_SOURCES
    slicelib.cpp
    floatTIFF.cpp
    cfpix.cpp
    ransubs.cpp
)

# Create the TEMSIM static library
add_library(temsim_lib STATIC ${TEMSIM_LIB_SOURCES})
target_include_directories(temsim_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${FFTW_INCLUDE_DIR})
target_link_libraries(temsim_lib PUBLIC ${FFTW_LIBRARIES} Threads::Threads)

# List of executables and their source files
set(EXECUTABLES
    atompot
    image
    mulslice
    pdb2xyz
    probe
    stemslic
    sumpix
)

# Build the executables
foreach(exec ${EXECUTABLES})
    add_executable(${exec} ${exec}.cpp)
    target_link_libraries(${exec} PRIVATE temsim_lib)
endforeach()

# Include additional source for the probe executable
if(TARGET probe)
    target_sources(probe PRIVATE probecmd.cpp)
endif()

# incostem executable (depends on FFTW and Threads)
add_executable(incostem incostemcmd.cpp incostem.cpp probe.cpp)
target_include_directories(incostem PRIVATE ${FFTW_INCLUDE_DIR})
target_link_libraries(incostem PRIVATE temsim_lib ${FFTW_LIBRARIES} Threads::Threads)

# Set CUDA compilation flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --generate-code arch=compute_80,code=sm_80")

# Build autoslic_cuda executable
add_executable(autoslic_cuda autosliccmd_cuda.cpp autoslic_cuda.cu probe.cpp cudaSlice.cu rfpix.cpp)
set_target_properties(autoslic_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_include_directories(autoslic_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(autoslic_cuda PRIVATE temsim_lib cuda cufft)

# Build autostem_cuda executable
add_executable(autostem_cuda autostemcmd_cuda.cpp autostem_cuda.cu cudaSlice.cu rfpix.cpp)
set_target_properties(autostem_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_include_directories(autostem_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(autostem_cuda PRIVATE temsim_lib cuda cufft)
