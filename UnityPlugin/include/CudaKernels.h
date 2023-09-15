#pragma once

#include <string>
#include <cuda_runtime.h>
#include <exception>
#include <stdexcept>

inline void cuda_error_throw() throw(std::bad_exception) {
#if DEBUG || _DEBUG
	if (cudaPeekAtLastError() != cudaSuccess) {
		throw std::runtime_error(cudaGetErrorString(cudaGetLastError()));
	}
	cudaDeviceSynchronize();
	if (cudaPeekAtLastError() != cudaSuccess) {
		throw std::runtime_error(cudaGetErrorString(cudaGetLastError()));
	}
#endif
}

inline bool cuda_error(std::string& _message) {
#if DEBUG || _DEBUG
	if (cudaPeekAtLastError() != cudaSuccess) {
		_message.assign(cudaGetErrorString(cudaGetLastError()));
		return true;
	}
	cudaDeviceSynchronize();
	if (cudaPeekAtLastError() != cudaSuccess) {
		_message.assign(cudaGetErrorString(cudaGetLastError()));
		return true;
	}
#endif
	return false;
}

void cuda_fill(int width, int height, float value, cudaSurfaceObject_t surface);
void cuda_splat_to_texture(int width, int height, float* rgb, cudaSurfaceObject_t surface);
