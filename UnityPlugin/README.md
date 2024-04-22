# Gaussian Splatting Unity Plugin

Unity plugin with original Cuda Rasterizer.

## Compilation

Install [Visual studio 2019](https://learn.microsoft.com/fr-fr/visualstudio/install/install-visual-studio?view=vs-2019) with c++ app development workload.

Install [Cuda 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive), don't forget to install visual studio integration, and update your nvidia driver if needed.

Install [Cmake](https://cmake.org/).

Launch this command.
```sh
cmake -G "Visual Studio 16" . -B build
cmake --build build --config Release -j 4
```

## Installation

Copy `build/gaussiansplatting.dll` to unity project in `Assets\GaussianSplattingPlugin\Plugins`.

## Usage

Here is the list of dll entries. for a concrete usage example, see unity project.

```csharp
//Init event value, send it with GL.IssuePluginEvent after SetNbPov and SetPovParameters
public const int INIT_EVENT = 0x0001;
//Draw event value, send it with GL.IssuePluginEvent after Preprocessed
public const int DRAW_EVENT = 0x0002;
//preprocess event value, send it with GL.IssuePluginEvent after SetDrawParameters
public const int PREPROCESS_EVENT = 0x0003;

// Entry point for GL.IssuePluginEvent
[DllImport("gaussiansplatting", EntryPoint = "GetRenderEventFunc")] public static extern IntPtr GetRenderEventFunc();

// Return true if api is ready
[DllImport("gaussiansplatting", EntryPoint = "IsAPIReady")] public static extern bool IsAPIReady();

//Get the last message from api
[DllImport("gaussiansplatting", EntryPoint = "GetLastMessage")] private static extern IntPtr _GetLastMessage();
static public string GetLastMessage() { return Marshal.PtrToStringAnsi(_GetLastMessage()); }

//Load the model, this call is blocking so it's better to start it in a thread.
[DllImport("gaussiansplatting", EntryPoint = "LoadModel")] public static extern bool LoadModel(string file);

//Import last loaded model to cuda return the internal id of the model
[DllImport("gaussiansplatting", EntryPoint = "CopyModelToCuda")] public static extern int CopyModelToCuda();

//Remove a model from cuda
[DllImport("gaussiansplatting", EntryPoint = "RemoveModelFromCuda")] public static extern bool RemoveModelFromCuda(int model);

//Set cropbox to crop a model
[DllImport("gaussiansplatting", EntryPoint = "SetModelCrop")] public static extern void SetModelCrop(int model, float[] box_min, float[] box_max);

//Get default cropbox value from a model
[DllImport("gaussiansplatting", EntryPoint = "GetModelCrop")] public static extern void GetModelCrop(int model, float[] box_min, float[] box_max);

//Set the model as active or not. if not active it will not be drawn (for unity mapping)
[DllImport("gaussiansplatting", EntryPoint = "SetActiveModel")] public static extern void SetActiveModel(int model, bool active);

//Get the total number of splats in cuda memory (just for informations)
[DllImport("gaussiansplatting", EntryPoint = "GetNbSplat")] public static extern int GetNbSplat();

//Create a new POV
[DllImport("gaussiansplatting", EntryPoint = "CreatePov")] public static extern int CreatePov();

//Remove a POV
[DllImport("gaussiansplatting", EntryPoint = "RemovePov")] public static extern void RemovePov(int pov);

//Set the POV parameters
[DllImport("gaussiansplatting", EntryPoint = "SetPovParameters")] public static extern void SetPovParameters(int pov, int width, int height);

//Initialize the POV
[DllImport("gaussiansplatting", EntryPoint = "IsInitialized")] public static extern bool IsInitialized(int pov);

//Get native pointer to created target texture for a pov. Use it with Texture2D.CreateExternalTexture
[DllImport("gaussiansplatting", EntryPoint = "GetTextureNativePointer")] public static extern IntPtr GetTextureNativePointer(int pov);

//Set the camera depth texture to be used to merge with gaussians
[DllImport("gaussiansplatting", EntryPoint = "SetCameraDepthTextureNativePointer")] public static extern void SetCameraDepthTextureNativePointer(int pov, IntPtr ptr);

//Get native pointer of depth texture returned by gaussian renderer
[DllImport("gaussiansplatting", EntryPoint = "GetDepthTextureNativePointer")] public static extern IntPtr GetDepthTextureNativePointer(int pov);

//Set draw parameter for a pov. proj matrix should be sent column first.
[DllImport("gaussiansplatting", EntryPoint = "SetDrawParameters")] public static extern void SetDrawParameters(int pov, int model, float[] position, float[] rotation, float[] scale, float[] proj, float fovy, float[] frustums);

//Check if a POV is preprocessed
[DllImport("gaussiansplatting", EntryPoint = "IsPreprocessed")] public static extern bool IsPreprocessed(int pov);
//Check if a POV is drawn
[DllImport("gaussiansplatting", EntryPoint = "IsDrawn")] public static extern bool IsDrawn(int pov);
```
