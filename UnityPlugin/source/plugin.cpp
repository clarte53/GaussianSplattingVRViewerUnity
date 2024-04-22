#include "plugin.h"
#include "PlatformBase.h"
#include "PluginAPI.h"

#include <string>
#include <sstream>

using namespace std;

static IUnityInterfaces* s_UnityInterfaces = NULL;
static IUnityGraphics* s_Graphics = NULL;
static PluginAPI* api = nullptr;

extern "C" UNITY_INTERFACE_EXPORT const char* UNITY_INTERFACE_API GetLastMessage() {
	if (api == nullptr) { return "NO API"; }
	return api->GetLastMessage();
}

extern "C" UNITY_INTERFACE_EXPORT bool UNITY_INTERFACE_API IsAPIReady() {
	return api != nullptr;
}

extern "C" UNITY_INTERFACE_EXPORT bool UNITY_INTERFACE_API LoadModel(const char* file) {
	if (api == nullptr) { return false; }
	return api->LoadModel(file);
}

extern "C" UNITY_INTERFACE_EXPORT int UNITY_INTERFACE_API CopyModelToCuda() {
	if (api == nullptr) { return 0; }
	return api->CopyModelToCuda();
}

extern "C" UNITY_INTERFACE_EXPORT bool UNITY_INTERFACE_API RemoveModelFromCuda(int model) {
	if (api == nullptr) { return false; }
	return api->RemoveModelFromCuda(model);
}

extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API SetActiveModel(int model, bool active) {
	if (api == nullptr) { return; }
	api->SetActiveModel(model, active);
}

extern "C" UNITY_INTERFACE_EXPORT int UNITY_INTERFACE_API CreatePov() {
	if (api == nullptr) { return 0; }
	return api->CreatePov();
}

extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API RemovePov(int pov) {
	if (api == nullptr) { return; }
	api->RemovePov(pov);
}

extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API SetPovParameters(int pov, int width, int height) {
	if (api == nullptr) { return; }
	api->SetPovParameters(pov, width, height);
}

extern "C" UNITY_INTERFACE_EXPORT bool UNITY_INTERFACE_API IsInitialized(int pov) {
	if (api == nullptr) { return false; }
	return api->IsInitialized(pov);
}

extern "C" UNITY_INTERFACE_EXPORT bool UNITY_INTERFACE_API IsDrawn(int pov) {
	if (api == nullptr) { return false; }
	return api->IsDrawn(pov);
}

extern "C" UNITY_INTERFACE_EXPORT bool UNITY_INTERFACE_API IsPreprocessed(int pov) {
	if (api == nullptr) { return false; }
	return api->IsPreprocessed(pov);
}

extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API SetDrawParameters(int pov, int model, float* position, float* rotation, float* scale, float* proj, float fovy, float* frustums) {
	if (api == nullptr) { return; }
	api->SetDrawParameters(pov, model, position, rotation, scale, proj, fovy, frustums);
}

extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API GetModelCrop(int model, float* box_min, float* box_max) {
	if (api == nullptr) { return; }
	api->splat.GetModelCrop(model, box_min, box_max);
}

extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API SetModelCrop(int model, float* box_min, float* box_max) {
	if (api == nullptr) { return; }
	api->splat.SetModelCrop(model, box_min, box_max);
}

extern "C" UNITY_INTERFACE_EXPORT int UNITY_INTERFACE_API GetNbSplat() {
	if (api == nullptr) { return 0; }
	return api->splat.GetNbSplat();
}

extern "C" UNITY_INTERFACE_EXPORT void* UNITY_INTERFACE_API GetTextureNativePointer(int pov) {
	if (api == nullptr) { return nullptr; }
	api->GetTextureNativePointer(pov);
}

extern "C" UNITY_INTERFACE_EXPORT void* UNITY_INTERFACE_API GetDepthTextureNativePointer(int pov) {
	if (api == nullptr) { return nullptr; }
	api->GetDepthTextureNativePointer(pov);
}

extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API SetCameraDepthTextureNativePointer(int pov, void* ptr) {
	if (api == nullptr) { return; }
	api->SetCameraDepthTextureNativePointer(pov, ptr);
}

//Render event callback, called by GL.IssuePluginEvent on render thread.
static void UNITY_INTERFACE_API OnRenderEvent(int eventID)
{
	if (api != nullptr) {
		api->OnRenderEvent(eventID);
	}
}


//Device event callback
static void UNITY_INTERFACE_API OnGraphicsDeviceEvent(UnityGfxDeviceEventType eventType)
{
	// Create graphics API implementation upon initialization
	if (eventType == UnityGfxDeviceEventType::kUnityGfxDeviceEventInitialize)
	{
		if (api != nullptr) { delete api; api = nullptr; }

		//Initialize api
		PluginAPI* t_api = nullptr;
		t_api = PluginAPI::Create(s_Graphics->GetRenderer(), s_UnityInterfaces);
		if (t_api != nullptr) {
			t_api->OnGraphicsDeviceEvent(eventType);
		}
		api = t_api;
		return;
	}

	//Other events are launched
	if (api != nullptr) {
		api->OnGraphicsDeviceEvent(eventType);
	}

	// Cleanup graphics API implementation upon shutdown
	if (eventType == UnityGfxDeviceEventType::kUnityGfxDeviceEventShutdown)
	{
		if (api != nullptr) { delete api; api = nullptr; }
	}
}

// Section to interface unity.
extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API UnityPluginLoad(IUnityInterfaces * unityInterfaces)
{
	s_UnityInterfaces = unityInterfaces;
	s_Graphics = s_UnityInterfaces->Get<IUnityGraphics>();
	s_Graphics->RegisterDeviceEventCallback(OnGraphicsDeviceEvent);

#if SUPPORT_VULKAN
	if (s_Graphics->GetRenderer() == kUnityGfxRendererNull)
	{
		extern void RenderAPI_Vulkan_OnPluginLoad(IUnityInterfaces*);
		RenderAPI_Vulkan_OnPluginLoad(unityInterfaces);
	}
#endif // SUPPORT_VULKAN

	// Run OnGraphicsDeviceEvent(initialize) manually on plugin load
	OnGraphicsDeviceEvent(UnityGfxDeviceEventType::kUnityGfxDeviceEventInitialize);
}

extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API UnityPluginUnload()
{
	s_Graphics->UnregisterDeviceEventCallback(OnGraphicsDeviceEvent);
}

extern "C" UNITY_INTERFACE_EXPORT UnityRenderingEvent UNITY_INTERFACE_API GetRenderEventFunc()
{
    return OnRenderEvent;
}
