#include "GaussianSplatting.h"
#include "GaussianSplatting.h"
#include "CudaKernels.h"
#include <cuda_runtime.h>
#include <rasterizer.h>
#include <fstream>
#include <string>
#include <sstream>
#include <iostream>

using namespace std;

typedef	Eigen::Matrix<int, 3, 1, Eigen::DontAlign> Vector3i;

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

template<typename T> float* append_cuda(float* cuda, size_t sz, vector<T>& data) {
	float* ncuda = nullptr;
	size_t snb = sizeof(T) * data.size();
	size_t size = sizeof(T) * sz;

	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&ncuda, size + snb));
	if (cuda != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(ncuda, cuda, size, cudaMemcpyDeviceToDevice)); }
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(((char*)ncuda) + size, data.data(), snb, cudaMemcpyHostToDevice));
	if (cuda != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree(cuda)); }
	return ncuda;
}

template<typename T> float* remove_cuda(float* cuda, size_t sz, size_t pos, size_t nb) {
	if (cuda == nullptr) { return nullptr; }

	float* ncuda = nullptr;
	size_t snb = sizeof(T) * nb;
	size_t spos = sizeof(T) * pos;
	size_t size = sizeof(T) * sz;
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&ncuda, size - snb));
	if (spos > 0) {
		CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(ncuda, cuda, spos, cudaMemcpyDeviceToDevice));
	}
	if (spos + snb < size) {
		CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(((char*)ncuda + spos), ((char*)cuda) + spos + snb, size - snb - spos, cudaMemcpyDeviceToDevice));
	}
	CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)cuda));
	return ncuda;
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

void GaussianSplattingRenderer::SetModelCrop(int model, float* box_min, float* box_max) {
	for (std::list<SplatModel>::iterator it = models.begin(); it != models.end(); ++it) {
		if (it->index == model) {
			it->_boxmin = Vector3f(box_min);
			it->_boxmax = Vector3f(box_max);
			break;
		}
	}
}

void GaussianSplattingRenderer::GetModelCrop(int model, float* box_min, float* box_max) {
	for (std::list<SplatModel>::iterator it = models.begin(); it != models.end(); ++it) {
		if (it->index == model) {
			box_min[0] = it->_scenemin.x();
			box_min[1] = it->_scenemin.y();
			box_min[2] = it->_scenemin.z();
			box_max[0] = it->_scenemax.x();
			box_max[1] = it->_scenemax.y();
			box_max[2] = it->_scenemax.z();
			break;
		}
	}
}

int GaussianSplattingRenderer::GetNbSplat() {
	return count;
}

void GaussianSplattingRenderer::Load(const char* file) {
	count_cpu = 0;
	
	// Load the PLY data (AoS) to the GPU (SoA)
	if (_sh_degree == 1)
	{
		count_cpu = loadPly<1>(file, pos, shs, opacity, scale, rot, _scenemin, _scenemax);
	}
	else if (_sh_degree == 2)
	{
		count_cpu = loadPly<2>(file, pos, shs, opacity, scale, rot, _scenemin, _scenemax);
	}
	else if (_sh_degree == 3)
	{
		count_cpu = loadPly<3>(file, pos, shs, opacity, scale, rot, _scenemin, _scenemax);
	}
}

int GaussianSplattingRenderer::CopyToCuda() {
	if (count_cpu == 0) {
		return 0;
	}

	const std::lock_guard<std::mutex> lock(cuda_mtx);

	//Register new model
	model_idx += 1;
	models.push_back({ model_idx, count_cpu, false, _scenemin, _scenemax, _scenemin, _scenemax });

	pos_cuda = append_cuda(pos_cuda, count, pos);
	rot_cuda = append_cuda(rot_cuda, count, rot);
	shs_cuda = append_cuda(shs_cuda, count, shs);
	opacity_cuda = append_cuda(opacity_cuda, count, opacity);
	scale_cuda = append_cuda(scale_cuda, count, scale);

	//set new size with the appened model
	count += count_cpu;

	//Working buffer or fixed data
	//can be fully reallocated
	if (background_cuda != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)background_cuda)); }
	if (rect_cuda != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)rect_cuda)); }
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&background_cuda, 3 * sizeof(float)));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&rect_cuda, 2 * count * sizeof(int)));

	bool white_bg = false;
	float bg[3] = { white_bg ? 1.f : 0.f, white_bg ? 1.f : 0.f, white_bg ? 1.f : 0.f };
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(background_cuda, bg, 3 * sizeof(float), cudaMemcpyHostToDevice));
	
	AllocateRenderContexts();

	//Update count and return new model index
	return model_idx;
}

void GaussianSplattingRenderer::RemoveModel(int model) {
	const std::lock_guard<std::mutex> lock(cuda_mtx);
	
	size_t start = 0;
	std::list<SplatModel>::iterator mit = models.end();
	for (std::list<SplatModel>::iterator it = models.begin(); it != models.end(); ++it) {
		if (it->index == model) {
			mit = it;
			break;
		}
		start += it->size;
	}

	if (mit != models.end()) {
		size_t size = mit->size;
		pos_cuda = remove_cuda<Pos>(pos_cuda, count, start, size);
		rot_cuda = remove_cuda<Rot>(rot_cuda, count, start, size);
		shs_cuda = remove_cuda<SHs<3>>(shs_cuda, count, start, size);
		opacity_cuda = remove_cuda<float>(opacity_cuda, count, start, size);
		scale_cuda = remove_cuda<Scale>(scale_cuda, count, start, size);

		count -= size;
		models.erase(mit);

		//Working buffer or fixed data
		//can be fully reallocated
		if (background_cuda != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)background_cuda)); }
		if (rect_cuda != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)rect_cuda)); }
		CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&background_cuda, 3 * sizeof(float)));
		CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&rect_cuda, 2 * count * sizeof(int)));

		bool white_bg = false;
		float bg[3] = { white_bg ? 1.f : 0.f, white_bg ? 1.f : 0.f, white_bg ? 1.f : 0.f };
		CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(background_cuda, bg, 3 * sizeof(float), cudaMemcpyHostToDevice));

		AllocateRenderContexts();

	} else {
		throw std::runtime_error("Model index not found.");
	}
}

void GaussianSplattingRenderer::CreateRenderContext(int idx) {

	const std::lock_guard<std::mutex> lock(cuda_mtx);

	//Resize the buffers
	geom[idx] = new AllocFuncBuffer;
	binning[idx] = new AllocFuncBuffer;
	img[idx] = new AllocFuncBuffer;
	renData[idx] = new RenderData;

	//Alloc
	geom[idx]->bufferFunc = resizeFunctional(&geom[idx]->ptr, geom[idx]->allocd);
	binning[idx]->bufferFunc = resizeFunctional(&binning[idx]->ptr, binning[idx]->allocd);
	img[idx]->bufferFunc = resizeFunctional(&img[idx]->ptr, img[idx]->allocd);

	//Alloc cuda ressource for view model
	AllocateRenderContexts();
}

void GaussianSplattingRenderer::RemoveRenderContext(int idx) {
	const std::lock_guard<std::mutex> lock(cuda_mtx);
	
	//freee cuda resources
	if (geom.at(idx)->ptr != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)geom.at(idx)->ptr)); }
	if (binning.at(idx)->ptr != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)binning.at(idx)->ptr)); }
	if (img.at(idx)->ptr != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)img.at(idx)->ptr)); }

	geom.erase(idx);
	binning.erase(idx);
	img.erase(idx);

	if (renData.at(idx)->view_cuda != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)renData.at(idx)->view_cuda)); }
	if (renData.at(idx)->proj_cuda != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)renData.at(idx)->proj_cuda)); }
	if (renData.at(idx)->model_sz != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)renData.at(idx)->model_sz)); }
	if (renData.at(idx)->model_active != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)renData.at(idx)->model_active)); }
	if (renData.at(idx)->cam_pos_cuda != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)renData.at(idx)->cam_pos_cuda)); }
	if (renData.at(idx)->boxmin != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)renData.at(idx)->boxmin)); }
	if (renData.at(idx)->boxmax != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)renData.at(idx)->boxmax)); }
	if (renData.at(idx)->frustums != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)renData.at(idx)->frustums)); }

	RenderData* data = renData.at(idx);
	renData.erase(idx);
	delete data;
}

void GaussianSplattingRenderer::AllocateRenderContexts() {
	size_t nb_models = models.size();
	for (auto kv: renData) {
		RenderData* data = kv.second;
		//reallocate only if needed
		if (data->nb_model_allocated != nb_models) {
			data->nb_model_allocated = nb_models;
			
			//free last allocated ressources
			if (data->view_cuda != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)(data->view_cuda))); }
			if (data->proj_cuda != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)(data->proj_cuda))); }
			if (data->model_sz != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)(data->model_sz))); }
			if (data->model_active != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)(data->model_active))); }
			if (data->cam_pos_cuda != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)(data->cam_pos_cuda))); }
			if (data->boxmin != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)(data->boxmin))); }
			if (data->boxmax != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)(data->boxmax))); }
			if (data->frustums != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)(data->frustums))); }

			// Create space for view parameters for each model
			CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&(data->view_cuda), sizeof(Matrix4f) * nb_models));
			CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&(data->proj_cuda), sizeof(Matrix4f) * nb_models));
			CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&(data->model_sz), sizeof(int) * nb_models));
			CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&(data->model_active), sizeof(int) * nb_models));
			CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&(data->cam_pos_cuda), 3 * sizeof(float) * nb_models));
			CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&(data->boxmin), 3 * sizeof(float) * nb_models));
			CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&(data->boxmax), 3 * sizeof(float) * nb_models));
			CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&(data->frustums), 6 * sizeof(float)));
		}
	}
}

void GaussianSplattingRenderer::SetActiveModel(int model, bool active) {
	for (SplatModel& m : models) {
		if (m.index == model) {
			m.active = active;
		}
	}
}

void GaussianSplattingRenderer::Preprocess(int context, const std::map<int, Matrix4f>& view_mat, const std::map<int, Matrix4f>& proj_mat, const std::map<int, Vector3f>& position, Vector6f frumstums, float fovy, int width, int height) {
	//view_mat.row(1) *= -1;
	//view_mat.row(2) *= -1;
	//proj_mat.row(1) *= -1;

	const std::lock_guard<std::mutex> lock(cuda_mtx);

	if (count == 0) { return; }
	
	float aspect_ratio = (float)width / (float)height;
	float tan_fovy = tan(fovy * 0.5f);
	float tan_fovx = tan_fovy * aspect_ratio;

	RenderData* rdata = renData.at(context);
	int nb_models = models.size();
	int midx = 0;
	for (const SplatModel& m : models) {
		int active = (m.active && view_mat.find(m.index) != view_mat.end()) ? 1 : 0;
		int msize = m.size;
		CUDA_SAFE_CALL(cudaMemcpy((char*)(rdata->model_sz) + midx * sizeof(int), &msize, sizeof(int), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy((char*)(rdata->model_active) + midx * sizeof(int), &active, sizeof(int), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy((char*)(rdata->boxmin) + midx * sizeof(float) * 3, m._boxmin.data(), sizeof(float) * 3, cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy((char*)(rdata->boxmax) + midx * sizeof(float) * 3, m._boxmax.data(), sizeof(float) * 3, cudaMemcpyHostToDevice));
		if (active == 1) {
			CUDA_SAFE_CALL(cudaMemcpy((char*)(rdata->view_cuda) + midx * sizeof(Matrix4f), view_mat.at(m.index).data(), sizeof(Matrix4f), cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL(cudaMemcpy((char*)(rdata->proj_cuda) + midx * sizeof(Matrix4f), proj_mat.at(m.index).data(), sizeof(Matrix4f), cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL(cudaMemcpy((char*)(rdata->cam_pos_cuda) + midx * sizeof(float) * 3, position.at(m.index).data(), sizeof(float) * 3, cudaMemcpyHostToDevice));
		}
		midx += 1;
	}
	CUDA_SAFE_CALL(cudaMemcpy((char*)(rdata->frustums), frumstums.data(), sizeof(float) * 6, cudaMemcpyHostToDevice));

	// Rasterize
	int* rects = _fastCulling ? rect_cuda : nullptr;
	rdata->num_rendered = CudaRasterizer::Rasterizer::forward_preprocess(
		geom.at(context)->bufferFunc,
		binning.at(context)->bufferFunc,
		img.at(context)->bufferFunc,
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
		rdata->view_cuda,
		rdata->proj_cuda,
		rdata->cam_pos_cuda,
		rdata->frustums,
		rdata->model_sz,
		rdata->model_active,
		nb_models,
		tan_fovx,
		tan_fovy,
		false,
		nullptr,
		rects,
		rdata->boxmin,
		rdata->boxmax);
}

void GaussianSplattingRenderer::Render(int context, float* image_cuda, float* depth_cuda, cudaSurfaceObject_t camera_depth_cuda, float fovy, int width, int height) {
	if (count > 0 && renData.at(context)->num_rendered > 0) {
		
		RenderData* rdata = renData.at(context);
		
		const std::lock_guard<std::mutex> lock(cuda_mtx);
		
		float aspect_ratio = (float)width / (float)height;
		float tan_fovy = tan(fovy * 0.5f);
		float tan_fovx = tan_fovy * aspect_ratio;

		int* rects = _fastCulling ? rect_cuda : nullptr;

		CudaRasterizer::Rasterizer::forward_render(
			geom.at(context)->bufferFunc,
			binning.at(context)->bufferFunc,
			img.at(context)->bufferFunc,
			count, _sh_degree, 16,
			background_cuda,
			camera_depth_cuda,
			width, height,
			pos_cuda,
			shs_cuda,
			nullptr,
			opacity_cuda,
			scale_cuda,
			_scalingModifier,
			rot_cuda,
			nullptr,
			rdata->view_cuda,
			rdata->proj_cuda,
			rdata->cam_pos_cuda,
			tan_fovx,
			tan_fovy,
			false,
			image_cuda,
			depth_cuda,
			nullptr,
			rects,
			rdata->boxmin,
			rdata->boxmax,
			rdata->num_rendered);
	} else {
		CUDA_SAFE_CALL(cudaMemset(image_cuda, 0, sizeof(float) * 4 * width * height));
		CUDA_SAFE_CALL(cudaMemset(depth_cuda, 0, sizeof(float) * width * height));
	}
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
	int lcount;
	ss >> dummy >> dummy >> lcount;

	while (std::getline(infile, buff))
		if (buff.compare("end_header") == 0)
			break;

	// Read all Gaussians at once (AoS)
	std::vector<RichPoint<D>> points(lcount);
	infile.read((char*)points.data(), lcount * sizeof(RichPoint<D>));

	// Resize our SoA data
	pos.resize(lcount);
	shs.resize(lcount);
	scales.resize(lcount);
	rot.resize(lcount);
	opacities.resize(lcount);

	// Gaussians are done training, they won't move anymore. Arrange
	// them according to 3D Morton order. This means better cache
	// behavior for reading Gaussians that end up in the same tile 
	// (close in 3D --> close in 2D).
	minn = Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);
	maxx = -minn;
	for (int i = 0; i < lcount; i++)
	{
		maxx = maxx.cwiseMax(points[i].pos);
		minn = minn.cwiseMin(points[i].pos);
	}
	std::vector<std::pair<uint64_t, int>> mapp(lcount);
	for (int i = 0; i < lcount; i++)
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
	for (int k = 0; k < lcount; k++)
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
	return lcount;
}
