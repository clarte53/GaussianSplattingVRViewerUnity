#pragma once

#include "Unity/IUnityInterface.h"
#include "Unity/IUnityGraphics.h"
#include "GaussianSplatting.h"

#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <map>

struct POV {
	int width = 0;
	int height = 0;
	std::map<int, Matrix4f> model_view_mat;
	std::map<int, Matrix4f> model_proj_view_mat;
	std::map<int, Vector3f> model_position;
	Vector6f frustums;
	float fovy;

	std::mutex event_mtx;
	
	//Convert this to state
	bool is_initialized = false;
	bool is_drawn = false;
	bool is_preprocessed = false;
	bool is_drawable = false;

	cudaGraphicsResource_t imageBufferCuda = nullptr;
	float* splatBufferCuda = nullptr;
	cudaGraphicsResource_t imageDepthBufferCuda = nullptr;
	float* splatDepthBufferCuda = nullptr;
	cudaGraphicsResource_t imageCameraDepthBufferCuda = nullptr;
	float* splatCameraDepthBufferCuda = nullptr;

	bool _interop_failed = false;
	std::vector<char> fallback_bytes;
	float* fallbackBufferCuda = nullptr;
	std::vector<char> fallback_depth_bytes;
	float* fallbackDepthBufferCuda = nullptr;

	virtual ~POV();
	virtual void* GetTextureNativePointer() = 0;
	virtual void* GetDepthTextureNativePointer() = 0;
	virtual void SetCameraDepthTextureNativePointer(void* ptr) = 0;
	virtual bool Init(std::string& message) = 0;
	void FreeCudaRessources();
	bool AllocSplatBuffer(std::string& message);
	bool AllocFallbackIfNeeded(std::string& message);
};

class PluginAPI {
protected:
	std::string _message = "";
	int _povidx = 0;
	std::map<int, std::shared_ptr<POV>> povs;
	int _device = 0;
	std::mutex pov_create_mtx;

public:
	GaussianSplattingRenderer splat;

public:
	virtual ~PluginAPI();
	const char* GetLastMessage();

	bool LoadModel(const char* file);
	int CopyModelToCuda();
	bool RemoveModelFromCuda(int model);
	void SetActiveModel(int model, bool active);
	
	int CreatePov();
	void RemovePov(int pov);
	void SetPovParameters(int pov, int width, int height);
	bool IsInitialized(int pov);

	void SetDrawParameters(int pov, int model, float* position, float* rotation, float* scale, float* proj, float fovy, float* frustums);
	bool IsDrawn(int pov);
	bool IsPreprocessed(int pov);

	void OnGraphicsDeviceEvent(UnityGfxDeviceEventType eventType);
	void OnRenderEvent(int eventID);

	void Preprocess();
	void Draw();
	void* GetTextureNativePointer(int pov);
	void* GetDepthTextureNativePointer(int pov);
	void SetCameraDepthTextureNativePointer(int pov, void* ptr);
	bool SetAndCheckCudaDevice();
	void InitPovs();

	virtual POV* CreatePOV() = 0;
	virtual void Init() = 0;

public:
	static PluginAPI* Create(UnityGfxRenderer s_DeviceType, IUnityInterfaces* s_UnityInterfaces);

	static const int INIT_EVENT = 0x0001;
	static const int DRAW_EVENT = 0x0002;
	static const int PREPROCESS_EVENT = 0x0003;
};
