# Unity Native Plugin Base

This project is a plugin for unity including CUDA. This can be the base project to create a unity plugins that use CUDA sharing gpu memory.

# compilation

Install Cuda 11.8, and visual studio 2019 c++.

It use cmake with visual studio 2019 to compile.

```sh
cmake -G "Visual Studio 16" . -B build
cmake --build build --config Release -j 4
```
