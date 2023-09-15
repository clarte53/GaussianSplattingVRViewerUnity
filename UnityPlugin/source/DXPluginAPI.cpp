#include "DXPluginAPI.h"
#include "CudaKernels.h"

#include <sstream>
#include <chrono>

using namespace std;
using namespace std::chrono;

inline bool dx_error(HRESULT hr, std::string& _message) {
#if DEBUG || _DEBUG
	if (FAILED(hr)) {
		_message.assign((stringstream()<<"DirectX ERROR: 0x"<<hex<<hr).str());
		return true;
	}
#endif
	return false;
}

DXPluginAPI::DXPluginAPI(IUnityInterfaces* interfaces) {
	IUnityGraphicsD3D11* d3d = interfaces->Get<IUnityGraphicsD3D11>();
	dxDevice = d3d->GetDevice();
}

DXPluginAPI::DXPOV::DXPOV(ID3D11Device* device) {
	dxDevice = device;
}

DXPluginAPI::DXPOV::~DXPOV() {
	//DirectX with Texture and cuda interop
	if (pTexture) { pTexture->Release(); }
	if (pShaderView) { pShaderView->Release(); }
}

DXPluginAPI::~DXPluginAPI() {
	povs.resize(0);
}

bool DXPluginAPI::DXPOV::Init(string& message) {
	//cuda interop
	if (pTexture) { pTexture->Release(); }
	if (pShaderView) { pShaderView->Release(); }
	POV::FreeCudaRessources();

	//Alloc a new splat buffer for results
	POV::AllocSplatBuffer(message);

	HRESULT hr;
	ID3D11DeviceContext* dxContext;
	dxDevice->GetImmediateContext(&dxContext);
	D3D11_TEXTURE2D_DESC desc = { width, height, 1, 1, DXGI_FORMAT_R32G32B32A32_FLOAT, {1,0}, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE, 0, 0 };
	hr = dxDevice->CreateTexture2D(&desc, nullptr, &pTexture); if (dx_error(hr, message)) { return false; }
	hr = dxDevice->CreateShaderResourceView(pTexture, nullptr, &pShaderView); if (dx_error(hr, message)) { return false; }

	if (cudaPeekAtLastError() != cudaSuccess) { message = cudaGetErrorString(cudaGetLastError()); return false; }
	cudaGraphicsD3D11RegisterResource(&imageBufferCuda, pTexture, cudaGraphicsRegisterFlagsSurfaceLoadStore);
	_interop_failed = !(cudaPeekAtLastError() == cudaSuccess);

	return POV::AllocFallbackIfNeeded(message);
}

void* DXPluginAPI::DXPOV::GetTextureNativePointer() {
	return pShaderView;
}

POV* DXPluginAPI::CreatePOV() {
	return new DXPOV(dxDevice);
}

bool DXPluginAPI::Init()
{
	//Get a cuda device for current DirectX device.
	HRESULT hr;
	_device = -1;
	int* devices = new int[1] { -1 };
	unsigned int nb_devices = 0;
	cudaD3D11GetDevices(&nb_devices, devices, 1, dxDevice, cudaD3D11DeviceListCurrentFrame);
	if (nb_devices > 0) {
		_device = devices[0];
	}
	delete[] devices;
	if (cuda_error(_message)) { return false; }

	if (nb_devices == 0) {
		_message = "No CUDA devices detected!";
		return false;
	}

	if (!PluginAPI::SetAndCheckCudaDevice()) { return false; }
	return PluginAPI::InitPovs();
}
