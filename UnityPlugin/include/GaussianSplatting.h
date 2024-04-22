#pragma once

#include <Eigen/Eigen>
#include <cuda_runtime.h>
#include <mutex>

typedef Eigen::Matrix<float, 3, 1, Eigen::DontAlign> Vector3f;
typedef Eigen::Matrix<float, 6, 1, Eigen::DontAlign> Vector6f;
typedef	Eigen::Matrix<float, 4, 4, Eigen::DontAlign, 4, 4> Matrix4f;

typedef Vector3f Pos;
template<int D> struct SHs { float shs[(D + 1) * (D + 1) * 3]; };
struct Scale { float scale[3]; };
struct Rot { float rot[4]; };

class GaussianSplattingRenderer {
	//TODO: create a parameters
	int _sh_degree = 3; //used when learning 3 is the default value
	bool _fastCulling = true;
	float _scalingModifier = 1.0;
	
	//Fix data (the model for cuda)
	int count = 0;
	float* pos_cuda = nullptr;
	float* rot_cuda = nullptr;
	float* scale_cuda = nullptr;
	float* opacity_cuda = nullptr;
	float* shs_cuda = nullptr;
	int* rect_cuda = nullptr;

	struct AllocFuncBuffer {
		size_t allocd = 0;
		void* ptr = nullptr;
		std::function<char* (size_t N)> bufferFunc;
	};

	std::map<int, AllocFuncBuffer*> geom;
	std::map<int, AllocFuncBuffer*> binning;
	std::map<int, AllocFuncBuffer*> img;

	//Changing data (the pov)
	float* background_cuda = nullptr;
	struct RenderData {
		int nb_model_allocated = 0;
		int num_rendered = 0;
		float* view_cuda = nullptr;
		float* proj_cuda = nullptr;
		float* cam_pos_cuda = nullptr;
		int* model_active = nullptr;
		int* model_sz = nullptr;
		float* boxmin = nullptr;
		float* boxmax = nullptr;
		float* frustums = nullptr;
	};

	std::map<int, RenderData*> renData;

	int model_idx = 0;

	struct SplatModel {
		int index;
		int size;
		bool active;
		Vector3f _boxmin, _boxmax, _scenemin, _scenemax;
	};

	std::list<SplatModel> models;
	Vector3f _scenemin, _scenemax;

	std::mutex cuda_mtx;

public:
	//Cpu version of the datas (for loading)
	int count_cpu;
	std::vector<Pos> pos;
	std::vector<Rot> rot;
	std::vector<Scale> scale;
	std::vector<float> opacity;
	std::vector<SHs<3>> shs;

public:
	void Load(const char* file) throw(std::bad_exception);
	int CopyToCuda();
	void RemoveModel(int model) throw(std::bad_exception);
	void SetActiveModel(int model, bool active);
	void CreateRenderContext(int idx);
	void RemoveRenderContext(int idx);
	void Preprocess(int context, const std::map<int, Matrix4f>& view_mat, const std::map<int, Matrix4f>& proj_mat, const std::map<int, Vector3f>& position, Vector6f frumstums, float fovy, int width, int height) throw(std::bad_exception);
	void Render(int context, float* image_cuda, float* depth_cuda, cudaSurfaceObject_t camera_depth_cuda, float fovy, int width, int height) throw(std::bad_exception);
	void Render(float* image_cuda, float* depth_cuda, Matrix4f view_mat, Matrix4f proj_mat, Vector3f position, Vector6f frumstums, float fovy, int width, int height) throw(std::bad_exception);
	void SetModelCrop(int model, float* box_min, float* box_max);
	void GetModelCrop(int model, float* box_min, float* box_max);
	int GetNbSplat();

private:
	void AllocateRenderContexts();
};
