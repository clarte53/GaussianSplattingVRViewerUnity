#pragma once

#include <Eigen/Eigen>
typedef Eigen::Matrix<float, 3, 1, Eigen::DontAlign> Vector3f;
typedef	Eigen::Matrix<float, 4, 4, Eigen::DontAlign, 4, 4> Matrix4f;

typedef Vector3f Pos;
template<int D> struct SHs { float shs[(D + 1) * (D + 1) * 3]; };
struct Scale { float scale[3]; };
struct Rot { float rot[4]; };

class GaussianSplattingRenderer {
	//TODO: create a parameters
	int _sh_degree = 3; //used when learning 3 is the default value
	bool _fastCulling = true;
	bool _cropping = true;
	float _scalingModifier = 1.0;
	
	//The crop box and scene limit
	Vector3f _boxmin, _boxmax, _scenemin, _scenemax;
	
	//Fix data (the model for cuda)
	int count;
	float* pos_cuda = nullptr;
	float* rot_cuda = nullptr;
	float* scale_cuda = nullptr;
	float* opacity_cuda = nullptr;
	float* shs_cuda = nullptr;
	int* rect_cuda = nullptr;

	size_t allocdGeom = 0, allocdBinning = 0, allocdImg = 0;
	void* geomPtr = nullptr, * binningPtr = nullptr, * imgPtr = nullptr;
	std::function<char* (size_t N)> geomBufferFunc, binningBufferFunc, imgBufferFunc;

	//Changing data (the pov)
	float* view_cuda = nullptr;
	float* proj_cuda = nullptr;
	float* cam_pos_cuda = nullptr;
	float* background_cuda = nullptr;

public:
	//Cpu version of the datas
	std::vector<Pos> pos;
	std::vector<Rot> rot;
	std::vector<Scale> scale;
	std::vector<float> opacity;
	std::vector<SHs<3>> shs;

public:
	void Load(const char* file) throw(std::bad_exception);
	void Render(float* image_cuda, Matrix4f view_mat, Matrix4f proj_mat, Vector3f position, float fovy, int width, int height) throw(std::bad_exception);
	void SetCrop(float* box_min, float* box_max);
	void GetSceneSize(float* scene_min, float* scene_max);
};
