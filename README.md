# Gaussian Splatting VR Viewer

At SIGGRAPH 2023 the paper "[**3D Gaussian Splatting for Real-Time Radiance Field Rendering**](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)" by Kerbl, Kopanas, Leimkühler, Drettakis has been published and features impressive rendering speeds compared to other nerf techniques.

Clarte has integrated [**Differential Gaussian Rasterization**](https://github.com/graphdeco-inria/diff-gaussian-rasterization/tree/main) in a Unity Native Plugin to create a OpenXR viewer of gaussian splatting learned models. This project is the original implementation of gaussian splatting renderer written in CUDA. The plugin also works without OpenXR to render gaussian splatting in Unity.

> A pre-compiled version of the OpenXR viewer for windows is available for testing. You can also compile from source or open the unity project.

![Screenshot](/screenshot.png)

Other people have implemented their own gaussian splatting renderers. For example, aras-p has implemented a renderer for Unity ([UnityGaussianSplatting](https://github.com/aras-p/UnityGaussianSplatting)), and the CVLab at EPFL has implemented a WebGL renderer ([gaussian-splatting-web](https://github.com/cvlab-epfl/gaussian-splatting-web)).

## Installation

### Hardware Requirements

- MS Windows VR Ready computer
- Minimal GPU: CUDA-ready GPU with Compute Capability 7.0+ (Geforce > 2060): [Check your card here](https://developer.nvidia.com/cuda-gpus)
- Recommended GPU: GeForce > 4070.

### Our test hardware was
  - CPU: Intel Core i7-11700K
  - RAM: 16Go
  - GPU: NVIDIA GeForce RTX 3060 Ti
  - VRAM: 8Go
  - Windows 10

### How to start

Download the latest version of the VR viewer [release](https://github.com/clarte53/GaussianSplattingVRViewerUnity/releases).

Connect your headset and set its runtime as the default OpenXR runtime.

Launch the executable `GaussianSplattingVRViewer.exe`, it will launch the VR application in OpenXR environment. You should see something like this.

![Screenshot clarte](screen_default.png)

Enjoy gaussian splatting in VR.

### Loading an other model

You can display your own model. For example, you can download the original [Pre-trained Models (14 GB)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip), generate a model with [Polycam service](https://poly.cam/gaussian-splatting/), or train your own model using the [reference toolkit](https://github.com/graphdeco-inria/gaussian-splatting)

To do so, copy a `point_cloud.ply` file, for example the one in `models/bicycle/point_cloud/iteration_30000` (in the pre-trained model zip), in the same folder as the windows executable `GaussianSplattingVRViewer.exe`, like in the image below.

![point_cloud](/point_cloud_ply.png)

Launch the executable `GaussianSplattingVRViewer.exe`, it will load the **point_cloud.ply** file instead of the default default demo model.

Enjoy gaussian splatting in VR again!

By default the rendering resolution of the splats is 50% of the headset native resolution in order to limit performance issues. That's why the images displayed may be a bit pixelized. The rendering resolution can be changed in the menu using a slider (see below).

## Usage

Press one `Grip` button fo the VR controllers to rotate and move the scene.

Press both `Grip` buttons to scale the scene. A line wil appear between the two controllers to materialize a ruler that can help scaling the model properly.

The `Joystick` on the left controller can be used to translate in the scene. For example, push the joystick forward to fly in the direction pointed by the controller.

The `Joystick` on the right controller can be used to turn left, right or 180°.

The `Menu` button on left controller is used to display a menu. In this menu, you can change the rendering resolution, watch the rendering speed and the number of splats of the model. A button to quit the application is also available.

Alternatively, the `escape` or `Q` keys can be pressed to quit the application.

## Use the plugin or compile from source

Start by cloning the whole project with recursive submodules.

```
git clone git@github.com:clarte53/GaussianSplattingVRViewerUnity.git --recursive
```

In order to use the plugin in unity see [Unity Project Readme](/GaussianSplattingVRViewer/README.md). The precompiled dll provided can be used immediately. A c# wrapper is also included in the unity project.

Alternatively, the dll can be compiled from the source. See [Plugin Compilation Readme](/UnityPlugin/README.md).

## Performances

On the sample "bicycle" scene, from the viewpoint depicted below, rendering 1024x1024 pixels on an NVIDIA GeForce RTX 3060 Ti gives to the following measurements:
  - Official SiBr viewer / OpenGL(FOV 60°): 12ms (80 FPS)
  - Unity / DirectX 11 (FOV 60°): 15ms (67 FPS)
  - Unity / OpenXR (2 eyes, FOV 90°): 38ms (26 FPS)

![pov](performance_pov.png)

In VR more splats are rendered because of the 90° fov instead of 60° and there is two 1024x1024 texture to render, 1 per eye.

![pov](performance_vr.png)

# Dependencies

Source used as submodule: [Differential Gaussian Rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization/tree/main).

Dependencies distributed as precompiled dlls: [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page), [gl3w](https://github.com/skaslev/gl3w), [glew](https://glew.sourceforge.net/).
