#pragma once
#include "Unity/IUnityInterface.h"
#include "Unity/IUnityGraphics.h"

extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API UnityPluginLoad(IUnityInterfaces * unityInterfaces);
extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API UnityPluginUnload();
extern "C" UNITY_INTERFACE_EXPORT UnityRenderingEvent UNITY_INTERFACE_API GetRenderEventFunc();

extern "C" UNITY_INTERFACE_EXPORT bool UNITY_INTERFACE_API IsAPIReady();
extern "C" UNITY_INTERFACE_EXPORT const char* UNITY_INTERFACE_API GetLastMessage();
extern "C" UNITY_INTERFACE_EXPORT bool UNITY_INTERFACE_API LoadModel(const char* file);
extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API SetNbPov(int nb_pov);
extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API SetPovParameters(int pov, int width, int height);
extern "C" UNITY_INTERFACE_EXPORT bool UNITY_INTERFACE_API IsInitialized();
extern "C" UNITY_INTERFACE_EXPORT void* UNITY_INTERFACE_API GetTextureNativePointer(int pov);
extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API SetDrawParameters(int pov, float* position, float* rotation, float* proj, float fovy, float* frustums);
extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API GetSceneSize(float* scene_min, float* scene_max);
extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API SetCrop(float* box_min, float* box_max);
extern "C" UNITY_INTERFACE_EXPORT bool UNITY_INTERFACE_API DrawSync();
extern "C" UNITY_INTERFACE_EXPORT bool UNITY_INTERFACE_API IsDrawn();

extern "C" UNITY_INTERFACE_EXPORT int UNITY_INTERFACE_API GetNbSplat();
