#pragma once

#include "PluginAPI.h"

#include <d3d11.h>
#include "Unity/IUnityGraphicsD3D11.h"

#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include <vector>

class DXPluginAPI : public PluginAPI {
public:
	DXPluginAPI(IUnityInterfaces* interfaces);
	virtual ~DXPluginAPI();
	virtual POV* CreatePOV() override;
	virtual bool Init() override;
private:
	ID3D11Device* dxDevice;
	int _device;

	struct DXPOV : public POV {
		ID3D11Device* dxDevice;
		ID3D11Texture2D* pTexture = nullptr;
		ID3D11ShaderResourceView* pShaderView = nullptr;

		DXPOV(ID3D11Device* device);
		virtual ~DXPOV();
		virtual bool Init(std::string& message) override;
		virtual void* GetTextureNativePointer() override;
	};
};
