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
//Draw event value, send it with GL.IssuePluginEvent after SetDrawParameters
public const int DRAW_EVENT = 0x0002;

// Entry point for GL.IssuePluginEvent
[DllImport("gaussiansplatting", EntryPoint = "GetRenderEventFunc")] public static extern System.IntPtr GetRenderEventFunc();

// Return true if api is ready
[DllImport("gaussiansplatting", EntryPoint = "IsAPIReady")] public static extern bool IsAPIReady();

//Get the last message from api
[DllImport("gaussiansplatting", EntryPoint = "GetLastMessage")] private static extern System.IntPtr _GetLastMessage();
static public string GetLastMessage() { return Marshal.PtrToStringAnsi(_GetLastMessage()); }

//Load the model, this call is blocking so it's better to start it in a thread.
[DllImport("gaussiansplatting", EntryPoint = "LoadModel")] public static extern bool LoadModel(string file);

//Set the number of POV to render each draw call (used for multiple camera)
[DllImport("gaussiansplatting", EntryPoint = "SetNbPov")] public static extern void SetNbPov(int nb_pov);

//Set parameter of a particular pov
[DllImport("gaussiansplatting", EntryPoint = "SetPovParameters")] public static extern void SetPovParameters(int pov, int width, int height);

//Wait this to be true after sending INIT_EVENT
[DllImport("gaussiansplatting", EntryPoint = "IsInitialized")] public static extern bool IsInitialized();

//Get native pointer to created target texture for a pov. Use it with Texture2D.CreateExternalTexture
[DllImport("gaussiansplatting", EntryPoint = "GetTextureNativePointer")] public static extern System.IntPtr GetTextureNativePointer(int pov);

//Set draw parameter for a pov. proj matrix should be sent column first with x inverted. (see matToFloat helper function below)
[DllImport("gaussiansplatting", EntryPoint = "SetDrawParameters")] public static extern void SetDrawParameters(int pov, float[] position, float[] rotation, float[] proj, float fovy, float[] frustums);

//Wait this to be true after sent DRAW_EVENT
[DllImport("gaussiansplatting", EntryPoint = "IsDrawn")] public static extern 
bool IsDrawn();

//Get nb splats of the model
[DllImport("gaussiansplatting", EntryPoint = "GetNbSplat")] public static extern int GetNbSplat();

//Set cropbox to crop splattings
[DllImport("gaussiansplatting", EntryPoint = "SetCrop")] public static extern void SetCrop(float[] box_min, float[] box_max);

//Get scene min max splattings positions
[DllImport("gaussiansplatting", EntryPoint = "GetSceneSize")] public static extern void GetSceneSize(float[] scene_min, float[] scene_max);

float[] matToFloat(Matrix4x4 mat)
    {
        return new float[16]
        {
            mat.m00, mat.m10, mat.m20, mat.m30,
            mat.m01, mat.m11, mat.m21, mat.m31,
            mat.m02, -mat.m12, mat.m22, mat.m32,
            mat.m03, mat.m13, mat.m23, mat.m33,
        };
    }
```
