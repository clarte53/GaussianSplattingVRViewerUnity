# Gaussian Splatting VR Viewer

At SIGGRAPH 2023 the paper "[**3D Gaussian Splatting for Real-Time Radiance Field Rendering**](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)" by Kerbl, Kopanas, Leimkühler, Drettakis had been released and had impressive speed result comparing to other nerf techniques.

Clarte integrate the *original CUDA implementation* of [**Differential Gaussian Rasterization**](https://github.com/graphdeco-inria/diff-gaussian-rasterization/tree/main) in a Unity Native Plugin to create a VR viewer of gaussian splatting learned models.

> For now, only a pre-compiled version for windows is available, source code will be published very soon.

![Screenshot](/screenshot.png)

Other people have done their own re-implementation of a gaussian splatting renderer for example, [UnityGaussianSplatting](https://github.com/aras-p/UnityGaussianSplatting) from aras-p or [gaussian-splatting-web](https://github.com/cvlab-epfl/gaussian-splatting-web) from cvlab-epfl.

## Installation

### Hardware Requirements

- VR Ready computer
- CUDA-ready GPU with Compute Capability 7.0+: [Check your card here](https://developer.nvidia.com/cuda-gpus)

### Our hardware was
  - CPU: Intel Core i7-11700K
  - RAM: 16Go
  - GPU: GeForce RTX 3060 Ti
  - VRAM: 8Go

### How to start

Download the last version of windows VR viewer [release](https://github.com/clarte53/GaussianSplattingVRViewerUnity/releases).

Download the pre-trained models [**Pre-trained Models (14 GB)**](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip) that you can found on the original git implementation [**GIT - 3D Gaussian Splatting for Real-Time Radiance Field Rendering**](https://github.com/graphdeco-inria/gaussian-splatting).

Copy a `point_cloud.ply` file, for example the one in `models/bicycle/point_cloud/iteration_30000`, near the windows executable `ClarteGaussianSpatting.exe`, like in the image below.

![point_cloud](/point_cloud_ply.png)

Connect your occulus headset in link or air link. You can also connect a steamvr compatible headset.

Launch the executable `ClarteGaussianSpatting.exe`, il will use Occulus software or SteamVR software automatically.

Enjoy the VR experiment.

## Usage

Press one `Grip` button to rotate the scene.

Press the two `Grip` buttons to scale and move the scene. A line appear between the two controllers to indicate the real distance between controllers. This distance is an helper for you to scale the world properly.

You can use `Joystick` (primary2DAxis) buttons to fly in the scene, for example, push the joystick toward and move the controller to fly in that direction. If you push the two `Joysticks` you will fly at higher speed.

Button `A` or `X` (PrimaryButton) is used to scale down the splat rendering texture size (by default at 50% of headset screen size) Button `B` or `Y` (SecondayButton) is used to scale it up.

If you don't use a Occulus Quest, you can found the key mapping of Unity [here](https://docs.unity3d.com/Manual/xr_input.html).

Press `escape` or `Q` on the keyboard to quit the application.

## Compile from source

We are working on a clean version of sources before publishing it, be free to play with the compiled version during this time.

# Dependencies

## Differential Gaussian Rasterization

Source used as submodule [Differential Gaussian Rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization/tree/main). This code is distributed under [this licence](/GaussianSplattingLicence.md).

```code
@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```

## Unity
[Unity](https://unity.com/) is used as main software development.

## Other dependencies

List of other dependencies: [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page), [gl3w](https://github.com/skaslev/gl3w), [glew](https://glew.sourceforge.net/).
