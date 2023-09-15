#pragma once

#include "Unity/IUnityInterface.h"
#include "Unity/IUnityGraphics.h"
#include "GaussianSplatting.h"

#include <cuda_runtime.h>
#include <string>
#include <vector>

struct POV {
	int width = 0;
	int height = 0;
	Matrix4f view_mat;
	Matrix4f proj_mat;
	Vector3f position;
	float fovy;

	cudaGraphicsResource_t imageBufferCuda = nullptr;
	float* splatBufferCuda = nullptr;

	bool _interop_failed = false;
	std::vector<char> fallback_bytes;
	float* fallbackBufferCuda = nullptr;

	virtual ~POV();
	virtual void* GetTextureNativePointer() = 0;
	virtual bool Init(std::string& message) = 0;
	void FreeCudaRessources();
	bool AllocSplatBuffer(std::string& message);
	bool AllocFallbackIfNeeded(std::string& message);
};

class PluginAPI {
protected:
	std::string _message = "";
	bool _is_initialized = false;
	bool _is_drawn = false;
	std::vector<std::shared_ptr<POV>> povs;
	int _device = 0;

public:
	GaussianSplattingRenderer splat;

public:
	virtual ~PluginAPI();
	const char* GetLastMessage();

	bool LoadModel(const char* file);
	void SetNbPov(int nb_pov);
	void SetPovParameters(int pov, int width, int height);
	bool IsInitialized();

	void SetDrawParameters(int pov, float* position, float* rotation, float* proj, float fovy, float* frustums);
	bool IsDrawn();

	void OnGraphicsDeviceEvent(UnityGfxDeviceEventType eventType);
	void OnRenderEvent(int eventID);

	bool Draw();
	void* GetTextureNativePointer(int pov);
	bool SetAndCheckCudaDevice();
	bool InitPovs();

	virtual POV* CreatePOV() = 0;
	virtual bool Init() = 0;

public:
	static PluginAPI* Create(UnityGfxRenderer s_DeviceType, IUnityInterfaces* s_UnityInterfaces);

	static const int INIT_EVENT = 0x0001;
	static const int DRAW_EVENT = 0x0002;
};
