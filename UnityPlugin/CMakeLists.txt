cmake_minimum_required(VERSION 3.18)

project(GaussianSplattingUnity
	VERSION 1.0
	DESCRIPTION "Gaussian Splatting for unity"
	LANGUAGES C CXX CUDA
)
set(UNITYNI_VERSION "${CMAKE_PROJECT_VERSION}")

if (NOT UNITYNI_DEPLOY)
	set(UNITYNI_VERSION "${UNITYNI_VERSION}dev")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

###############################################################################
# Build type and C++ compiler setup
###############################################################################

# Set a default configuration if none was specified
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "No release type specified. Setting to 'Release'.")
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif()

if (MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_CRT_SECURE_NO_WARNINGS")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP24")
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)

###############################################################################
# CUDA compiler setup
###############################################################################

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

if (MSVC)
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=/bigobj")
else()
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-Wno-float-conversion")
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fno-strict-aliasing")
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fPIC")
endif()
list(APPEND CUDA_NVCC_FLAGS "--extended-lambda")
list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")
list(APPEND CUDA_NVCC_FLAGS "--use_fast_math")

###############################################################################
# Dependencies
###############################################################################

find_package(OpenGL REQUIRED)

if (MSVC)
  list(APPEND UNITYNI_INCLUDE_DIRECTORIES "dependencies/gl3w")
  list(APPEND UNITYNI_SOURCES "dependencies/gl3w/GL/gl3w.c")
  list(APPEND UNITYNI_LIBRARIES opengl32)
  
  list(APPEND UNITYNI_INCLUDE_DIRECTORIES "dependencies/glew/glew-2.0.0/include")
  list(APPEND UNITYNI_LINK_DIRECTORIES "dependencies/glew/glew-2.0.0/lib64")
  list(APPEND UNITYNI_LIBRARIES glew32 glew32s glew32d glew32sd)
else()
  find_package(GLEW REQUIRED)
  list(APPEND UNITYNI_INCLUDE_DIRECTORIES ${GLEW_INCLUDE_DIRS})
endif()

add_subdirectory("dependencies/diff-gaussian-rasterization")
list(APPEND UNITYNI_LIBRARIES CudaRasterizer)

list(APPEND UNITYNI_INCLUDE_DIRECTORIES "dependencies/eigen/include")

###############################################################################
# Library
###############################################################################

list(APPEND UNITYNI_DEFINITIONS -DUNITYNI_VERSION="${UNITYNI_VERSION}")
list(APPEND UNITYNI_INCLUDE_DIRECTORIES "include")

# Add unity headers
file (GLOB UNITY_HEADERS "include/Unity/*.h")
list(APPEND UNITYNI_HEADERS ${UNITY_HEADERS})

if (NOT MSVC)
	list(APPEND UNITYNI_LIBRARIES ${CMAKE_DL_LIBS})
endif()

list(APPEND UNITYNI_HEADERS
)

list(APPEND UNITYNI_SOURCES
  ${UNITYNI_HEADERS}
  include/PlatformBase.h
  include/plugin.h
  include/PluginAPI.h
  include/GLPluginAPI.h
  include/DXPluginAPI.h
  include/CudaKernels.h
  include/GaussianSplatting.h
  
  source/plugin.cpp
  source/PluginAPI.cpp
  source/GLPluginAPI.cpp
  source/DXPluginAPI.cpp
  source/CudaKernels.cu
  source/GaussianSplatting.cpp
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR})

get_filename_component(CUDA_COMPILER_BIN "${CMAKE_CUDA_COMPILER}" DIRECTORY)
get_filename_component(CUDA_DIR "${CUDA_COMPILER_BIN}" DIRECTORY)
set(CUDA_INCLUDE "${CUDA_DIR}/include")
set(PROJECT_NAME gaussiansplatting)

add_library(${PROJECT_NAME} SHARED ${UNITYNI_SOURCES})
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(${PROJECT_NAME} PUBLIC ${UNITYNI_DEFINITIONS})
target_compile_options(${PROJECT_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
target_include_directories(${PROJECT_NAME} PUBLIC ${UNITYNI_INCLUDE_DIRECTORIES})
target_link_directories(${PROJECT_NAME} PUBLIC ${UNITYNI_LINK_DIRECTORIES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${UNITYNI_LIBRARIES})

# Copy shared libraries to the binary directory as needed
if (MSVC)
	file(GLOB CUDA_DLLS "${CUDA_COMPILER_BIN}/cudart64*.dll")
	if (CUDA_DLLS)
		add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CUDA_DLLS} "${CMAKE_CURRENT_BINARY_DIR}" COMMAND_EXPAND_LISTS)
	endif()
endif()
