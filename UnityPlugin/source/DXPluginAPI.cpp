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
	{
		//Allocate main texture
		D3D11_TEXTURE2D_DESC desc = { width, height, 1, 1, DXGI_FORMAT_R32G32B32A32_FLOAT, {1,0}, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE, 0, 0 };
		hr = dxDevice->CreateTexture2D(&desc, nullptr, &pTexture); if (dx_error(hr, message)) { return false; }
		hr = dxDevice->CreateShaderResourceView(pTexture, nullptr, &pShaderView); if (dx_error(hr, message)) { return false; }

		cudaGraphicsD3D11RegisterResource(&imageBufferCuda, pTexture, cudaGraphicsRegisterFlagsSurfaceLoadStore);
		_interop_failed = CUDA_ERROR(message);
	}

	if (!_interop_failed) {
		//Allocate depth texture
		D3D11_TEXTURE2D_DESC desc = { width, height, 1, 1, DXGI_FORMAT_R32_FLOAT, {1,0}, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE, 0, 0 };
		hr = dxDevice->CreateTexture2D(&desc, nullptr, &pDepthTexture); if (dx_error(hr, message)) { return false; }
		hr = dxDevice->CreateShaderResourceView(pDepthTexture, nullptr, &pDepthShaderView); if (dx_error(hr, message)) { return false; }

		cudaGraphicsD3D11RegisterResource(&imageDepthBufferCuda, pDepthTexture, cudaGraphicsRegisterFlagsSurfaceLoadStore);
		_interop_failed = CUDA_ERROR(message);
	}

	if (!_interop_failed) {
		//Map camera depth texture to cuda
		if (cudaPeekAtLastError() != cudaSuccess) { message = cudaGetErrorString(cudaGetLastError()); return false; }
		cudaGraphicsD3D11RegisterResource(&imageCameraDepthBufferCuda, pCameraDepthTexture, cudaGraphicsRegisterFlagsSurfaceLoadStore);
		_interop_failed = CUDA_ERROR(message);
	}

	return POV::AllocFallbackIfNeeded(message);
}

void* DXPluginAPI::DXPOV::GetTextureNativePointer() {
	return pShaderView;
}

void* DXPluginAPI::DXPOV::GetDepthTextureNativePointer() {
	return pDepthShaderView;
}

void DXPluginAPI::DXPOV::SetCameraDepthTextureNativePointer(void* ptr) {
	pCameraDepthTexture = (ID3D11Texture2D*)ptr;
}

POV* DXPluginAPI::CreatePOV() {
	return new DXPOV(dxDevice);
}

void DXPluginAPI::Init()
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
	if (CUDA_ERROR(_message)) { return; }

	if (nb_devices == 0) {
		_message = "No CUDA devices detected!";
		return;
	}

	if (!PluginAPI::SetAndCheckCudaDevice()) { return; }
	PluginAPI::InitPovs();
}
