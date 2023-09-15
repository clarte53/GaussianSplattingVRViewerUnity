#pragma once
#include "PluginAPI.h"

#include "GL/gl3w.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector>
#include <string>

class GLPluginAPI : public PluginAPI {
public:
	GLPluginAPI();
	virtual ~GLPluginAPI();
	virtual POV* CreatePOV() override;
	virtual bool Init() override;
private:

	struct GLPOV : public POV {
		GLuint imageBuffer = 0;
		virtual ~GLPOV();
		virtual bool Init(std::string& message) override;
		virtual void* GetTextureNativePointer() override;
	};
};
