cmake_minimum_required(VERSION 3.10)
project(temsim_gpu LANGUAGES CXX)

# Default to Release build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Use C++11 standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenMP if available
find_package(OpenMP)
set(OPENMP_LIB "")
if(OpenMP_CXX_FOUND)
    set(OPENMP_LIB OpenMP::OpenMP_CXX)
endif()

# Option to specify FFTW directory
set(FFTW_DIR "" CACHE PATH "Path to FFTW installation")

# Find FFTW headers and libraries
find_path(FFTW_INCLUDE_DIR fftw3.h
    HINTS ${FFTW_DIR}/include ${FFTW_DIR}
    PATH_SUFFIXES include
    DOC "FFTW include directory")

find_library(FFTW_LIBRARY fftw3f
    HINTS ${FFTW_DIR}/lib ${FFTW_DIR}/lib64 ${FFTW_DIR}
    PATH_SUFFIXES lib lib64
    DOC "FFTW single-precision library")

find_library(FFTW_THREADS_LIBRARY fftw3f_threads
    HINTS ${FFTW_DIR}/lib ${FFTW_DIR}/lib64 ${FFTW_DIR}
    PATH_SUFFIXES lib lib64
    DOC "FFTW single-precision threads library")

if(FFTW_INCLUDE_DIR AND FFTW_LIBRARY AND FFTW_THREADS_LIBRARY)
    set(FFTW_LIBRARIES ${FFTW_THREADS_LIBRARY} ${FFTW_LIBRARY})
else()
    message(FATAL_ERROR "Could not find FFTW libraries (fftw3f and fftw3f_threads).")
endif()

# Find Threads package
find_package(Threads REQUIRED)

# TEMSIM library sources
set(TEMSIM_LIB_SOURCES
    slicelib.cpp
    floatTIFF.cpp
    cfpix.cpp
    ransubs.cpp
)

# Create TEMSIM static library
add_library(temsim_lib STATIC ${TEMSIM_LIB_SOURCES})
target_include_directories(temsim_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${FFTW_INCLUDE_DIR}
)
target_link_libraries(temsim_lib PUBLIC
    ${FFTW_LIBRARIES}
    Threads::Threads
)

# Executables that do not need FFTW
set(EXECUTABLES
    atompot
    image
    mulslice
    pdb2xyz
    probe
    stemslic
    sumpix
)

foreach(exec ${EXECUTABLES})
    add_executable(${exec} ${exec}.cpp)
    target_link_libraries(${exec} PRIVATE temsim_lib)
endforeach()

# Add additional source to probe executable
if(TARGET probe)
    target_sources(probe PRIVATE probecmd.cpp)
endif()

# incostem executable
add_executable(incostem incostemcmd.cpp incostem.cpp probe.cpp)
target_include_directories(incostem PRIVATE ${FFTW_INCLUDE_DIR})
target_link_libraries(incostem PRIVATE temsim_lib ${FFTW_LIBRARIES} Threads::Threads)
set_target_properties(incostem PROPERTIES CXX_STANDARD 11)

# Executables with OpenMP
foreach(exec_name IN ITEMS autoslic autostem)
    if(${exec_name} STREQUAL "autoslic")
        set(SOURCES autosliccmd.cpp autoslic.cpp probe.cpp rfpix.cpp)
    elseif(${exec_name} STREQUAL "autostem")
        set(SOURCES autostemcmd.cpp autostem.cpp rfpix.cpp)
    endif()
    add_executable(${exec_name} ${SOURCES})
    target_include_directories(${exec_name} PRIVATE ${FFTW_INCLUDE_DIR})
    target_link_libraries(${exec_name} PRIVATE temsim_lib ${OPENMP_LIB} Threads::Threads)
    if(OpenMP_CXX_FOUND)
        target_compile_options(${exec_name} PRIVATE ${OpenMP_CXX_FLAGS})
    endif()
endforeach()
