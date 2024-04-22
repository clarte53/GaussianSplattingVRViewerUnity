#pragma once
#include "Unity/IUnityInterface.h"
#include "Unity/IUnityGraphics.h"

extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API UnityPluginLoad(IUnityInterfaces * unityInterfaces);
extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API UnityPluginUnload();
extern "C" UNITY_INTERFACE_EXPORT UnityRenderingEvent UNITY_INTERFACE_API GetRenderEventFunc();

extern "C" UNITY_INTERFACE_EXPORT bool UNITY_INTERFACE_API IsAPIReady();
extern "C" UNITY_INTERFACE_EXPORT const char* UNITY_INTERFACE_API GetLastMessage();
extern "C" UNITY_INTERFACE_EXPORT int UNITY_INTERFACE_API GetNbSplat();

//Load a model to cpu, call CopyModelToCuda to create a model index and draw it
extern "C" UNITY_INTERFACE_EXPORT bool UNITY_INTERFACE_API LoadModel(const char* file);

//Multi model management
extern "C" UNITY_INTERFACE_EXPORT int UNITY_INTERFACE_API CopyModelToCuda();
extern "C" UNITY_INTERFACE_EXPORT bool UNITY_INTERFACE_API RemoveModelFromCuda(int model);
extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API SetActiveModel(int model, bool active);
extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API SetModelCrop(int model, float* box_min, float* box_max);
extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API GetModelCrop(int model, float* box_min, float* box_max);

//Multi POV management
extern "C" UNITY_INTERFACE_EXPORT int UNITY_INTERFACE_API CreatePov();
extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API RemovePov(int pov);
extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API SetPovParameters(int pov, int width, int height);
extern "C" UNITY_INTERFACE_EXPORT bool UNITY_INTERFACE_API IsInitialized(int pov);
extern "C" UNITY_INTERFACE_EXPORT void* UNITY_INTERFACE_API GetTextureNativePointer(int pov);
extern "C" UNITY_INTERFACE_EXPORT void* UNITY_INTERFACE_API GetDepthTextureNativePointer(int pov);

//Pov rendering
extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API SetCameraDepthTextureNativePointer(int pov, void* ptr);
extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API SetDrawParameters(int pov, int model, float* position, float* rotation, float* scale, float* proj, float fovy, float* frustums);
extern "C" UNITY_INTERFACE_EXPORT bool UNITY_INTERFACE_API IsDrawn(int pov);
extern "C" UNITY_INTERFACE_EXPORT bool UNITY_INTERFACE_API IsPreprocessed(int pov);
