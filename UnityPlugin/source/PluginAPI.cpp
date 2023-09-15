#include "PluginAPI.h"
#include "GLPluginAPI.h"
#include "DXPluginAPI.h"
#include "CudaKernels.h"

#include <sstream>
#include <chrono>

using namespace std;
using namespace std::chrono;

typedef Eigen::Quaternion<float> Quaternionf;

PluginAPI::~PluginAPI() {
}

const char* PluginAPI::GetLastMessage() {
	return _message.c_str();
}

bool PluginAPI::LoadModel(const char* file) {
	_is_initialized = false;
	if (file == nullptr) { _message = "filepath cannot be null"; return false; }
	try {
		splat.Load(file);
	}
	catch (std::bad_exception ex) {
		_message = ex.what();
		return false;
	}
	return true;
}

void PluginAPI::SetNbPov(int nb_pov) {
	_is_initialized = false;
	povs.clear();
	for (int i = 0; i < nb_pov; ++i) {
		povs.push_back(shared_ptr<POV>(CreatePOV()));
	}
}
void* PluginAPI::GetTextureNativePointer(int pov) {
	return povs.at(pov)->GetTextureNativePointer();
}

void PluginAPI::SetPovParameters(int pov, int width, int height) {
	povs.at(pov)->width = width;
	povs.at(pov)->height = height;
}

bool PluginAPI::IsInitialized() { return _is_initialized; }

void PluginAPI::SetDrawParameters(int pov, float* position, float* rotation, float* proj, float fovy, float* frustums) {
	_is_drawn = false;
	float h = (float)povs.at(pov)->height;
	float w = (float)povs.at(pov)->width;

	//Use given proj
	Matrix4f proj_mat(proj);

	//Create rotation
	Quaternionf q(rotation[3], rotation[0], rotation[1], rotation[2]);
	Eigen::Matrix<float, 3, 3, 0, 3, 3> s = q.toRotationMatrix();
	Matrix4f rotmat;
	rotmat <<
		s(0, 0), s(0, 1), s(0, 2), 0,
		s(1, 0), s(1, 1), s(1, 2), 0,
		s(2, 0), s(2, 1), s(2, 2), 0,
		0, 0, 0, 1;
	
	//Create translation
	Vector3f pos(position[0], position[1], position[2]);
	Matrix4f posmat;
	posmat.setIdentity();
	posmat(0, 3) = pos.x();
	posmat(1, 3) = pos.y();
	posmat(2, 3) = pos.z();

	//Create transform and view matrix
	Matrix4f transform = posmat * rotmat;
	Matrix4f view_mat = transform.inverse();

	//Create view proj mat
	Matrix4f proj_view_mat = Matrix4f(proj_mat * view_mat);

	//Update Data
	povs.at(pov)->view_mat = view_mat;
	povs.at(pov)->proj_mat = proj_view_mat;
	povs.at(pov)->position = pos;
	povs.at(pov)->fovy = fovy;
}

bool PluginAPI::Draw() {
	long long ts = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

	int _nb_pov = povs.size();
	for (int i = 0; i < _nb_pov; ++i) {

		shared_ptr<POV>& pov = povs.at(i);
		cudaArray_t m_mapped_array = {};
		cudaSurfaceObject_t m_surface = {};

		float* image_cuda = pov->splatBufferCuda;
		if (!pov->_interop_failed) {

			size_t bytes;
			cudaGraphicsMapResources(1, &pov->imageBufferCuda); if (cuda_error(_message)) { return false; }
			cudaGraphicsSubResourceGetMappedArray(&m_mapped_array, pov->imageBufferCuda, 0, 0); if (cuda_error(_message)) { return false; }

			struct cudaResourceDesc resource_desc;
			memset(&resource_desc, 0, sizeof(resource_desc));
			resource_desc.resType = cudaResourceTypeArray;
			resource_desc.res.array.array = m_mapped_array;

			cudaCreateSurfaceObject(&m_surface, &resource_desc);  if (cuda_error(_message)) { return false; }
		}
		else {
			image_cuda = pov->fallbackBufferCuda;
		}


		try {
			splat.Render(image_cuda, pov->view_mat, pov->proj_mat, pov->position, pov->fovy, pov->width, pov->height);
		}
		catch (std::bad_exception ex) {
			_message = ex.what();
			return false;
		}

		if (!povs.at(i)->_interop_failed) {
			cuda_splat_to_texture(pov->width, pov->height, image_cuda, m_surface); if (cuda_error(_message)) { return false; }
			cudaDeviceSynchronize(); if (cuda_error(_message)) { return false; }
			cudaDestroySurfaceObject(m_surface); if (cuda_error(_message)) { return false; }
			cudaGraphicsUnmapResources(1, &povs.at(i)->imageBufferCuda); if (cuda_error(_message)) { return false; }
		} else {
			return false;
			//TODO: !!!
		}
	}

	long long after_ts = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
	_message = (stringstream() << "Last Time: " << (after_ts - ts) << "ms").str();

	return true;
}

bool PluginAPI::IsDrawn() { return _is_drawn; }

POV::~POV() {
	FreeCudaRessources();
}

void POV::FreeCudaRessources() {
	if (splatBufferCuda) { cudaFree(splatBufferCuda); splatBufferCuda = nullptr; }
	if (imageBufferCuda) { cudaGraphicsUnregisterResource(imageBufferCuda); imageBufferCuda = nullptr; }
	if (fallbackBufferCuda) { cudaFree(&fallbackBufferCuda); fallbackBufferCuda = nullptr; }
}

bool POV::AllocSplatBuffer(std::string& message) {
	//Alloc cuda buffer for splatting result
	cudaMalloc(&splatBufferCuda, width * height * 3 * sizeof(float)); if (cuda_error(message)) { return false; }
	return true;
}

bool POV::AllocFallbackIfNeeded(string& message) {
	//If interop failed alloc a cuda buffer
	if (_interop_failed) {
		message = cudaGetErrorString(cudaGetLastError());
		fallback_bytes.resize(width * height * 3 * sizeof(float));
		cudaMalloc(&fallbackBufferCuda, fallback_bytes.size());
	}
	else {
		//reset last error (it's a cudaSuccess)
		cudaGetLastError();
	}
	return true;
}

bool PluginAPI::SetAndCheckCudaDevice() {
	cudaSetDevice(_device); if (cuda_error(_message)) { return false; }
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, _device); if (cuda_error(_message)) { return false; }

	if (prop.major < 7) {
		_message = "Sorry, need at least compute capability 7.0+!";
		return false;
	}

	return true;
}

bool PluginAPI::InitPovs() {
	for (int i = 0; i < povs.size(); ++i) {
		auto pov = povs.at(i);
		if (!pov->Init(_message)) { return false; }
	}
	return true;
}

void PluginAPI::OnGraphicsDeviceEvent(UnityGfxDeviceEventType eventType) {
	_message = (stringstream() << "GLPluginAPI::OnGraphicsDeviceEvent(): " << (int)(eventType)).str();
}

void PluginAPI::OnRenderEvent(int eventID) {
	switch (eventID)
	{
	case PluginAPI::INIT_EVENT:
		_is_initialized = Init();
		break;
	case PluginAPI::DRAW_EVENT:
		_is_drawn = Draw();
		break;
	default:
		_message = (stringstream() << "GLPluginAPI::OnRenderEvent(): Unknown event Id " << eventID).str();
		break;
	}
}

PluginAPI* PluginAPI::Create(UnityGfxRenderer s_DeviceType, IUnityInterfaces* s_UnityInterfaces) {
	switch (s_DeviceType)
	{
	case UnityGfxRenderer::kUnityGfxRendererD3D11:
		return new DXPluginAPI(s_UnityInterfaces);
		break;
	case UnityGfxRenderer::kUnityGfxRendererGCM:
		break;
	case UnityGfxRenderer::kUnityGfxRendererNull:
		break;
	case UnityGfxRenderer::kUnityGfxRendererOpenGLES20:
		break;
	case UnityGfxRenderer::kUnityGfxRendererOpenGLES30:
		break;
	case UnityGfxRenderer::kUnityGfxRendererGXM:
		break;
	case UnityGfxRenderer::kUnityGfxRendererPS4:
		break;
	case UnityGfxRenderer::kUnityGfxRendererXboxOne:
		break;
	case UnityGfxRenderer::kUnityGfxRendererMetal:
		break;
	case UnityGfxRenderer::kUnityGfxRendererOpenGLCore:
		return new GLPluginAPI();
		break;
	case UnityGfxRenderer::kUnityGfxRendererD3D12:
		break;
	case UnityGfxRenderer::kUnityGfxRendererVulkan:
		break;
	case UnityGfxRenderer::kUnityGfxRendererNvn:
		break;
	case UnityGfxRenderer::kUnityGfxRendererXboxOneD3D12:
		break;
	default:
		break;
	}
	return nullptr;
}
