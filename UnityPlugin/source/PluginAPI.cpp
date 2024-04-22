#define _USE_MATH_DEFINES

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
	povs.clear();
}

const char* PluginAPI::GetLastMessage() {
	return _message.c_str();
}

bool PluginAPI::LoadModel(const char* file) {
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

int PluginAPI::CopyModelToCuda() {
	int model_num = 0;
	try {
		model_num = splat.CopyToCuda();
	}
	catch (std::bad_exception ex) {
		_message = ex.what();
	}
	return model_num;
}

bool PluginAPI::RemoveModelFromCuda(int model) {
	try {
		splat.RemoveModel(model);
	}
	catch (std::bad_exception ex) {
		_message = ex.what();
		return false;
	}
	return true;
}

int PluginAPI::CreatePov() {
	const std::lock_guard<std::mutex> lock(pov_create_mtx);
	_povidx += 1;
	povs[_povidx] = shared_ptr<POV>(CreatePOV());
	splat.CreateRenderContext(_povidx);
	return _povidx;
}

void PluginAPI::RemovePov(int pov) {
	const std::lock_guard<std::mutex> lock(pov_create_mtx);
	povs.erase(pov);
	splat.RemoveRenderContext(pov);
}

void* PluginAPI::GetTextureNativePointer(int pov) {
	return povs.at(pov)->GetTextureNativePointer();
}

void* PluginAPI::GetDepthTextureNativePointer(int pov) {
	return povs.at(pov)->GetDepthTextureNativePointer();
}

void PluginAPI::SetCameraDepthTextureNativePointer(int pov, void* ptr) {
	povs.at(pov)->SetCameraDepthTextureNativePointer(ptr);
}

void PluginAPI::SetPovParameters(int pov, int width, int height) {
	povs.at(pov)->width = width;
	povs.at(pov)->height = height;
}

bool PluginAPI::IsInitialized(int pov) { return povs.at(pov)->is_initialized; }

void PluginAPI::SetDrawParameters(int pov, int model, float* position, float* rotation, float* scale, float* proj, float fovy, float* frustums) {
	{
		const std::lock_guard<std::mutex> lock(povs.at(pov)->event_mtx);
		povs.at(pov)->is_drawable = false;
		povs.at(pov)->is_preprocessed = false;
		povs.at(pov)->is_drawn = false;
	}

	float h = (float)povs.at(pov)->height;
	float w = (float)povs.at(pov)->width;

	//Use given proj
	proj[9] = -proj[9];
	Matrix4f proj_mat(proj);

	//Create rotation
	Quaternionf q = Quaternionf(Eigen::AngleAxisf(M_PI, Vector3f::UnitZ())) * Quaternionf(rotation[3], rotation[0], -rotation[1], -rotation[2]);
	Eigen::Matrix<float, 3, 3, 0, 3, 3> s = q.toRotationMatrix();
	Matrix4f rotmat;
	rotmat <<
		s(0, 0), s(0, 1), s(0, 2), 0,
		s(1, 0), s(1, 1), s(1, 2), 0,
		s(2, 0), s(2, 1), s(2, 2), 0,
		0, 0, 0, 1;

	Matrix4f scalemat;
	scalemat <<
		scale[0], 0, 0, 0,
		0, scale[1], 0, 0,
		0, 0, scale[2], 0,
		0, 0, 0, 1;

	//Create translation
	Vector3f pos(position[0] * scale[0], -position[1] * scale[1], position[2] * scale[2]);
	Matrix4f posmat;
	posmat.setIdentity();
	posmat(0, 3) = pos.x();
	posmat(1, 3) = pos.y();
	posmat(2, 3) = pos.z();

	//Create transform and view matrix
	Matrix4f transform = posmat * rotmat;
	Matrix4f view_mat =  transform.inverse() * scalemat;

	//Create view proj mat
	Matrix4f proj_view_mat = proj_mat * view_mat;

	//Update Dataq
	povs.at(pov)->model_view_mat[model] = view_mat;
	povs.at(pov)->model_proj_view_mat[model] = proj_view_mat;
	povs.at(pov)->model_position[model] = pos;
	povs.at(pov)->frustums = Vector6f(frustums);
	povs.at(pov)->fovy = fovy;
	povs.at(pov)->is_drawable = true;
}

void PluginAPI::SetActiveModel(int model, bool active) {
	splat.SetActiveModel(model, active);
}

void PluginAPI::Preprocess() {
	for (auto kv: povs) {
		shared_ptr<POV>& pov = kv.second;
		const std::lock_guard<std::mutex> lock(pov->event_mtx);
		if (pov->is_initialized && pov->is_drawable && !pov->is_preprocessed) {
			try {
				splat.Preprocess(kv.first, pov->model_view_mat, pov->model_proj_view_mat, pov->model_position, pov->frustums, pov->fovy, pov->width, pov->height);
				pov->is_preprocessed = true;
			}
			catch (std::exception ex) {
				_message = ex.what();
				pov->is_preprocessed = false;
			}
		}
	}
}

void PluginAPI::Draw() {
	int _nb_pov = povs.size();

	cudaArray_t* m_mapped_array = new cudaArray_t[_nb_pov];
	cudaSurfaceObject_t* m_surface = new cudaSurfaceObject_t[_nb_pov];

	cudaArray_t* m_depth_mapped_array = new cudaArray_t[_nb_pov];
	cudaSurfaceObject_t* m_depth_surface = new cudaSurfaceObject_t[_nb_pov];

	cudaArray_t* m_camera_depth_mapped_array = new cudaArray_t[_nb_pov];
	cudaSurfaceObject_t* m_camera_depth_surface = new cudaSurfaceObject_t[_nb_pov];

	//MAP CUDA RESSOURCE to TEXTURE
	int contextidx = 0;
	for (auto kv : povs) {
		shared_ptr<POV>& pov = kv.second;
		const std::lock_guard<std::mutex> lock(pov->event_mtx);
		if (pov->is_initialized && pov->is_drawable && pov->is_preprocessed && !pov->is_drawn) {
			bool isdrawn = false;
			if (!pov->_interop_failed) {
				//Map image
				cudaGraphicsMapResources(1, &pov->imageBufferCuda); if (CUDA_ERROR(_message)) { pov->is_drawn = false; continue; }
				cudaGraphicsSubResourceGetMappedArray(&(m_mapped_array[contextidx]), pov->imageBufferCuda, 0, 0); if (CUDA_ERROR(_message)) { pov->is_drawn = false; continue; }

				struct cudaResourceDesc resource_desc;
				memset(&resource_desc, 0, sizeof(resource_desc));
				resource_desc.resType = cudaResourceTypeArray;
				resource_desc.res.array.array = m_mapped_array[contextidx];

				cudaCreateSurfaceObject(&(m_surface[contextidx]), &resource_desc);  if (CUDA_ERROR(_message)) { pov->is_drawn = false; continue; }

				//Map depth
				cudaGraphicsMapResources(1, &pov->imageDepthBufferCuda); if (CUDA_ERROR(_message)) { pov->is_drawn = false; continue; }
				cudaGraphicsSubResourceGetMappedArray(&(m_depth_mapped_array[contextidx]), pov->imageDepthBufferCuda, 0, 0); if (CUDA_ERROR(_message)) { pov->is_drawn = false; continue; }

				struct cudaResourceDesc depth_resource_desc;
				memset(&depth_resource_desc, 0, sizeof(depth_resource_desc));
				depth_resource_desc.resType = cudaResourceTypeArray;
				depth_resource_desc.res.array.array = m_depth_mapped_array[contextidx];

				cudaCreateSurfaceObject(&(m_depth_surface[contextidx]), &depth_resource_desc);  if (CUDA_ERROR(_message)) { pov->is_drawn = false; continue; }

				//Map camera depth
				cudaGraphicsMapResources(1, &pov->imageCameraDepthBufferCuda); if (CUDA_ERROR(_message)) { pov->is_drawn = false; continue; }
				cudaGraphicsSubResourceGetMappedArray(&(m_camera_depth_mapped_array[contextidx]), pov->imageCameraDepthBufferCuda, 0, 0); if (CUDA_ERROR(_message)) { pov->is_drawn = false; continue; }

				struct cudaResourceDesc camera_depth_resource_desc;
				memset(&camera_depth_resource_desc, 0, sizeof(camera_depth_resource_desc));
				camera_depth_resource_desc.resType = cudaResourceTypeArray;
				camera_depth_resource_desc.res.array.array = m_camera_depth_mapped_array[contextidx];

				cudaCreateSurfaceObject(&(m_camera_depth_surface[contextidx]), &camera_depth_resource_desc);  if (CUDA_ERROR(_message)) { pov->is_drawn = false; continue; }
			}

			float* image_cuda = pov->splatBufferCuda;
			float* depth_cuda = pov->splatDepthBufferCuda;
			if (pov->_interop_failed) {
				//TODO: set warning for performance but do it with fallback
				image_cuda = pov->fallbackBufferCuda;
				depth_cuda = pov->fallbackDepthBufferCuda;
			}

			try {
				splat.Render(kv.first, image_cuda, depth_cuda, m_camera_depth_surface[contextidx], pov->fovy, pov->width, pov->height);
			}
			catch (std::exception ex) {
				_message = ex.what();
				pov->is_drawn = false;
				pov->is_drawable = false;
				continue;
			}

			if (!pov->_interop_failed) {
				//Copy image & depth
				cuda_splat_to_texture(pov->width, pov->height, 4, image_cuda, m_surface[contextidx]); if (CUDA_ERROR(_message)) { pov->is_drawn = false; continue; }
				cuda_splat_to_texture(pov->width, pov->height, 1, depth_cuda, m_depth_surface[contextidx]); if (CUDA_ERROR(_message)) { pov->is_drawn = false; continue; }
				//cuda_copy_depth_kernel(pov->width, pov->height, m_camera_depth_surface, m_depth_surface); if (CUDA_ERROR(_message)) { pov->is_drawn = false; continue; }
				isdrawn = true;
			} else {
				//TODO: set warning for performance but do it with fallback
				pov->is_drawn = false;
				pov->is_drawable = false;
				continue;
			}

			//SYNC CUDA
			cudaDeviceSynchronize(); if (CUDA_ERROR(_message)) { pov->is_drawn = false; continue; }

			if (!pov->_interop_failed) {
				cudaDestroySurfaceObject(m_surface[contextidx]); if (CUDA_ERROR(_message)) { pov->is_drawn = false; continue; }
				cudaGraphicsUnmapResources(1, &pov->imageBufferCuda); if (CUDA_ERROR(_message)) { pov->is_drawn = false; continue; }

				cudaDestroySurfaceObject(m_depth_surface[contextidx]); if (CUDA_ERROR(_message)) { pov->is_drawn = false; continue; }
				cudaGraphicsUnmapResources(1, &pov->imageDepthBufferCuda); if (CUDA_ERROR(_message)) { pov->is_drawn = false; continue; }

				cudaDestroySurfaceObject(m_camera_depth_surface[contextidx]); if (CUDA_ERROR(_message)) { pov->is_drawn = false; continue; }
				cudaGraphicsUnmapResources(1, &pov->imageCameraDepthBufferCuda); if (CUDA_ERROR(_message)) { pov->is_drawn = false; continue; }
			}

			//unset is_drawable to wait next frame
			pov->is_drawn = isdrawn;
			pov->is_drawable = false;
		}

		++contextidx;
	}


	delete[] m_mapped_array;
	delete[] m_surface;
	delete[] m_depth_mapped_array;
	delete[] m_depth_surface;
	delete[] m_camera_depth_mapped_array;
	delete[] m_camera_depth_surface;
}

bool PluginAPI::IsDrawn(int pov) {
	const std::lock_guard<std::mutex> lock(povs.at(pov)->event_mtx);
	return povs.at(pov)->is_drawn;
}
bool PluginAPI::IsPreprocessed(int pov) {
	const std::lock_guard<std::mutex> lock(povs.at(pov)->event_mtx);
	return povs.at(pov)->is_preprocessed;
}

POV::~POV() {
	FreeCudaRessources();
}

void POV::FreeCudaRessources() {
	if (splatBufferCuda) { cudaFree(splatBufferCuda); splatBufferCuda = nullptr; }
	if (splatDepthBufferCuda) { cudaFree(splatDepthBufferCuda); splatDepthBufferCuda = nullptr; }
	if (imageBufferCuda) { cudaGraphicsUnregisterResource(imageBufferCuda); imageBufferCuda = nullptr; }
	if (imageDepthBufferCuda) { cudaGraphicsUnregisterResource(imageDepthBufferCuda); imageDepthBufferCuda = nullptr; }
	if (fallbackBufferCuda) { cudaFree(&fallbackBufferCuda); fallbackBufferCuda = nullptr; }
	if (fallbackDepthBufferCuda) { cudaFree(&fallbackDepthBufferCuda); fallbackDepthBufferCuda = nullptr; }
}

bool POV::AllocSplatBuffer(std::string& message) {
	//Alloc cuda buffer for splatting result
	cudaMalloc(&splatBufferCuda, width * height * 4 * sizeof(float)); if (CUDA_ERROR(message)) { return false; }
	cudaMalloc(&splatDepthBufferCuda, width * height * 1 * sizeof(float)); if (CUDA_ERROR(message)) { return false; }
	return true;
}

bool POV::AllocFallbackIfNeeded(string& message) {
	//If interop failed alloc a cuda buffer
	if (_interop_failed) {
		message = cudaGetErrorString(cudaGetLastError());
		fallback_bytes.resize(width * height * 4 * sizeof(float));
		cudaMalloc(&fallbackBufferCuda, fallback_bytes.size());
		fallback_depth_bytes.resize(width * height * 1 * sizeof(float));
		cudaMalloc(&fallbackDepthBufferCuda, fallback_depth_bytes.size());
	}
	else {
		//reset last error (it's a cudaSuccess)
		cudaGetLastError();
	}
	return true;
}

bool PluginAPI::SetAndCheckCudaDevice() {
	cudaSetDevice(_device); if (CUDA_ERROR(_message)) { return false; }
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, _device); if (CUDA_ERROR(_message)) { return false; }

	if (prop.major < 7) {
		_message = "Sorry, need at least compute capability 7.0+!";
		return false;
	}

	return true;
}

void PluginAPI::InitPovs() {
	for (auto kv : povs) {
		if (!kv.second->is_initialized) {
			kv.second->is_initialized = kv.second->Init(_message);
		}
	}
}

void PluginAPI::OnGraphicsDeviceEvent(UnityGfxDeviceEventType eventType) {
	_message = (stringstream() << "GLPluginAPI::OnGraphicsDeviceEvent(): " << (int)(eventType)).str();
}

void PluginAPI::OnRenderEvent(int eventID) {
	switch (eventID)
	{
	case PluginAPI::INIT_EVENT:
		{
			const std::lock_guard<std::mutex> lock(pov_create_mtx);
			Init();
		}
		break;
	case PluginAPI::PREPROCESS_EVENT:
		Preprocess();
		break;
	case PluginAPI::DRAW_EVENT:
		Draw();
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
