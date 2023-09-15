#include "GLPluginAPI.h"
#include "CudaKernels.h"
#include "PlatformBase.h"

#if SUPPORT_OPENGL_CORE
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#endif

#include <sstream>

using namespace std;

inline bool gl_error(const char* func, int line, std::string& _message) {
#if DEBUG || _DEBUG
	GLenum err = glGetError();
	if (err != GL_NO_ERROR) {
		_message.assign((stringstream()<<func<<"::"<<line<< " OpenglError: 0x" << hex << err).str());
		return true;
	}
#endif
	return false;
}


GLPluginAPI::GLPluginAPI() {
#if UNITY_WIN && SUPPORT_OPENGL_CORE
	gl3wInit();
#endif
}

GLPluginAPI::GLPOV::~GLPOV() {
	if (imageBuffer) { glDeleteTextures(1, &imageBuffer); }
}

GLPluginAPI::~GLPluginAPI() {
}

bool GLPluginAPI::GLPOV::Init(string& message) {
	//cuda interop
	if (imageBuffer) { glDeleteTextures(1, &imageBuffer); }
	POV::FreeCudaRessources();

	//Alloc a new splat buffer for results
	POV::AllocSplatBuffer(message);

	//Alloc Texture for final result
	glGenTextures(1, &imageBuffer);
	glBindTexture(GL_TEXTURE_2D, imageBuffer);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, width, 0, GL_RGBA, GL_FLOAT, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

	if (cudaPeekAtLastError() != cudaSuccess) { message = cudaGetErrorString(cudaGetLastError()); return false; }
	cudaGraphicsGLRegisterImage(&imageBufferCuda, imageBuffer, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
	_interop_failed = !(cudaPeekAtLastError() == cudaSuccess);

	return POV::AllocFallbackIfNeeded(message);
}

void* GLPluginAPI::GLPOV::GetTextureNativePointer() {
	return (void*)imageBuffer;
}

POV* GLPluginAPI::CreatePOV() {
	return new GLPOV;
}

bool GLPluginAPI::Init()
{
	int num_devices;
	cudaGetDeviceCount(&num_devices); if (cuda_error(_message)) { return false; }

	_device = 0;
	if (_device >= num_devices) {
		_message = "No CUDA devices detected!";
		return false;
	}

	if (!PluginAPI::SetAndCheckCudaDevice()) { return false; }
	return PluginAPI::InitPovs();
}
