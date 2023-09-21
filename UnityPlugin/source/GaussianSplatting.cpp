#include "GaussianSplatting.h"
#include "CudaKernels.h"
#include <cuda_runtime.h>
#include <rasterizer.h>
#include <fstream>
#include <string>
#include <sstream>

using namespace std;

typedef	Eigen::Matrix<int, 3, 1, Eigen::DontAlign> Vector3i;

#define CUDA_SAFE_CALL_ALWAYS(A) A; cuda_error_throw();
#define CUDA_SAFE_CALL(A) A; cuda_error_throw();

inline float sigmoid(const float m1) { return 1.0f / (1.0f + exp(-m1)); }

inline std::function<char* (size_t N)> resizeFunctional(void** ptr, size_t& S) {
	auto lambda = [ptr, &S](size_t N) {
		if (N > S)
		{
			if (*ptr)
				CUDA_SAFE_CALL(cudaFree(*ptr));
			CUDA_SAFE_CALL(cudaMalloc(ptr, 2 * N));
			S = 2 * N;
		}
		return reinterpret_cast<char*>(*ptr);
	};
	return lambda;
}

//Gaussian Splatting data structure
template<int D>
struct RichPoint
{
	Pos pos;
	float n[3];
	SHs<D> shs;
	float opacity;
	Scale scale;
	Rot rot;
};

template<int D> int loadPly(const char* filename, std::vector<Pos>& pos, std::vector<SHs<3>>& shs, std::vector<float>& opacities, std::vector<Scale>& scales, std::vector<Rot>& rot, Vector3f& minn, Vector3f& maxx) throw(std::bad_exception);

void GaussianSplattingRenderer::SetCrop(float* box_min, float* box_max) {
	_boxmin = Vector3f(box_min);
	_boxmax = Vector3f(box_max);
}

void GaussianSplattingRenderer::GetSceneSize(float* scene_min, float* scene_max) {
	scene_min[0] = _scenemin.x();
	scene_min[1] = _scenemin.y();
	scene_min[2] = _scenemin.z();
	scene_max[0] = _scenemax.x();
	scene_max[1] = _scenemax.y();
	scene_max[2] = _scenemax.z();
}

void GaussianSplattingRenderer::Load(const char* file) {
	// Load the PLY data (AoS) to the GPU (SoA)
	if (_sh_degree == 1)
	{
		count = loadPly<1>(file, pos, shs, opacity, scale, rot, _scenemin, _scenemax);
	}
	else if (_sh_degree == 2)
	{
		count = loadPly<2>(file, pos, shs, opacity, scale, rot, _scenemin, _scenemax);
	}
	else if (_sh_degree == 3)
	{
		count = loadPly<3>(file, pos, shs, opacity, scale, rot, _scenemin, _scenemax);
	}

	_boxmin = _scenemin;
	_boxmax = _scenemax;
	
	int P = count;
	
	//free last allocated ressources
	if (pos_cuda != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)pos_cuda)); }
	if (rot_cuda != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)rot_cuda)); }
	if (shs_cuda != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)shs_cuda)); }
	if (opacity_cuda != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)opacity_cuda)); }
	if (scale_cuda != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)scale_cuda)); }

	// Allocate and fill the GPU data
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&pos_cuda, sizeof(Pos) * P));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(pos_cuda, pos.data(), sizeof(Pos) * P, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&rot_cuda, sizeof(Rot) * P));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(rot_cuda, rot.data(), sizeof(Rot) * P, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&shs_cuda, sizeof(SHs<3>) * P));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(shs_cuda, shs.data(), sizeof(SHs<3>) * P, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&opacity_cuda, sizeof(float) * P));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(opacity_cuda, opacity.data(), sizeof(float) * P, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&scale_cuda, sizeof(Scale) * P));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(scale_cuda, scale.data(), sizeof(Scale) * P, cudaMemcpyHostToDevice));

	//free last allocated ressources
	if (view_cuda != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)view_cuda)); }
	if (proj_cuda != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)proj_cuda)); }
	if (cam_pos_cuda != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)cam_pos_cuda)); }
	if (background_cuda != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)background_cuda)); }
	if (rect_cuda != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)rect_cuda)); }

	// Create space for view parameters
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&view_cuda, sizeof(Matrix4f)));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&proj_cuda, sizeof(Matrix4f)));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&cam_pos_cuda, 3 * sizeof(float)));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&background_cuda, 3 * sizeof(float)));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&rect_cuda, 2 * P * sizeof(int)));
	
	bool white_bg = false;
	float bg[3] = { white_bg ? 1.f : 0.f, white_bg ? 1.f : 0.f, white_bg ? 1.f : 0.f };
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(background_cuda, bg, 3 * sizeof(float), cudaMemcpyHostToDevice));

	geomBufferFunc = resizeFunctional(&geomPtr, allocdGeom);
	binningBufferFunc = resizeFunctional(&binningPtr, allocdBinning);
	imgBufferFunc = resizeFunctional(&imgPtr, allocdImg);
}

void GaussianSplattingRenderer::Render(float* image_cuda, Matrix4f view_mat, Matrix4f proj_mat, Vector3f position, float fovy, int width, int height) {
	//view_mat.row(1) *= -1;
	//view_mat.row(2) *= -1;
	//proj_mat.row(1) *= -1;
	
	float aspect_ratio = (float)width / (float)height;
	float tan_fovy = tan(fovy * 0.5f);
	float tan_fovx = tan_fovy * aspect_ratio;

	CUDA_SAFE_CALL(cudaMemcpy(view_cuda, view_mat.data(), sizeof(Matrix4f), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(proj_cuda, proj_mat.data(), sizeof(Matrix4f), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(cam_pos_cuda, &position, sizeof(float) * 3, cudaMemcpyHostToDevice));
	
	// Rasterize
	int* rects = _fastCulling ? rect_cuda : nullptr;
	float* boxmin = _cropping ? (float*)&_boxmin : nullptr;
	float* boxmax = _cropping ? (float*)&_boxmax : nullptr;
	CudaRasterizer::Rasterizer::forward(
		geomBufferFunc,
		binningBufferFunc,
		imgBufferFunc,
		count, _sh_degree, 16,
		background_cuda,
		width, height,
		pos_cuda,
		shs_cuda,
		nullptr,
		opacity_cuda,
		scale_cuda,
		_scalingModifier,
		rot_cuda,
		nullptr,
		view_cuda,
		proj_cuda,
		cam_pos_cuda,
		tan_fovx,
		tan_fovy,
		false,
		image_cuda,
		nullptr,
		rects,
		boxmin,
		boxmax
	);
}

// Load the Gaussians from the given file.
template<int D>
int loadPly(const char* filename,
	std::vector<Pos>& pos,
	std::vector<SHs<3>>& shs,
	std::vector<float>& opacities,
	std::vector<Scale>& scales,
	std::vector<Rot>& rot,
	Vector3f& minn,
	Vector3f& maxx)
{
	std::ifstream infile(filename, std::ios_base::binary);

	if (!infile.good())
		throw std::runtime_error((stringstream() << "Unable to find model's PLY file, attempted:\n" << filename).str());

	// "Parse" header (it has to be a specific format anyway)
	std::string buff;
	std::getline(infile, buff);
	std::getline(infile, buff);

	std::string dummy;
	std::getline(infile, buff);
	std::stringstream ss(buff);
	int count;
	ss >> dummy >> dummy >> count;

	while (std::getline(infile, buff))
		if (buff.compare("end_header") == 0)
			break;

	// Read all Gaussians at once (AoS)
	std::vector<RichPoint<D>> points(count);
	infile.read((char*)points.data(), count * sizeof(RichPoint<D>));

	// Resize our SoA data
	pos.resize(count);
	shs.resize(count);
	scales.resize(count);
	rot.resize(count);
	opacities.resize(count);

	// Gaussians are done training, they won't move anymore. Arrange
	// them according to 3D Morton order. This means better cache
	// behavior for reading Gaussians that end up in the same tile 
	// (close in 3D --> close in 2D).
	minn = Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);
	maxx = -minn;
	for (int i = 0; i < count; i++)
	{
		maxx = maxx.cwiseMax(points[i].pos);
		minn = minn.cwiseMin(points[i].pos);
	}
	std::vector<std::pair<uint64_t, int>> mapp(count);
	for (int i = 0; i < count; i++)
	{
		Vector3f rel = (points[i].pos - minn).array() / (maxx - minn).array();
		Vector3f scaled = ((float((1 << 21) - 1)) * rel);
		Vector3i xyz = scaled.cast<int>();

		uint64_t code = 0;
		for (int i = 0; i < 21; i++) {
			code |= ((uint64_t(xyz.x() & (1 << i))) << (2 * i + 0));
			code |= ((uint64_t(xyz.y() & (1 << i))) << (2 * i + 1));
			code |= ((uint64_t(xyz.z() & (1 << i))) << (2 * i + 2));
		}

		mapp[i].first = code;
		mapp[i].second = i;
	}
	auto sorter = [](const std::pair < uint64_t, int>& a, const std::pair < uint64_t, int>& b) {
		return a.first < b.first;
	};
	std::sort(mapp.begin(), mapp.end(), sorter);

	// Move data from AoS to SoA
	int SH_N = (D + 1) * (D + 1);
	for (int k = 0; k < count; k++)
	{
		int i = mapp[k].second;
		pos[k] = points[i].pos;

		// Normalize quaternion
		float length2 = 0;
		for (int j = 0; j < 4; j++)
			length2 += points[i].rot.rot[j] * points[i].rot.rot[j];
		float length = sqrt(length2);
		for (int j = 0; j < 4; j++)
			rot[k].rot[j] = points[i].rot.rot[j] / length;

		// Exponentiate scale
		for (int j = 0; j < 3; j++)
			scales[k].scale[j] = exp(points[i].scale.scale[j]);

		// Activate alpha
		opacities[k] = sigmoid(points[i].opacity);

		shs[k].shs[0] = points[i].shs.shs[0];
		shs[k].shs[1] = points[i].shs.shs[1];
		shs[k].shs[2] = points[i].shs.shs[2];
		for (int j = 1; j < SH_N; j++)
		{
			shs[k].shs[j * 3 + 0] = points[i].shs.shs[(j - 1) + 3];
			shs[k].shs[j * 3 + 1] = points[i].shs.shs[(j - 1) + SH_N + 2];
			shs[k].shs[j * 3 + 2] = points[i].shs.shs[(j - 1) + 2 * SH_N + 1];
		}
	}
	return count;
}
