using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.XR;

public class GaussianSplatting : MonoBehaviour
{
	public enum State
	{
		ERROR = -1,
		DISABLED,
		INITIALIZATION,
		RENDERING,
		PAUSE,
	}

	public interface Observer
	{
		public void OnStateChanged(GaussianSplatting gs, State state);
	}

	public class Native
	{
		public const int INIT_EVENT = 0x0001;
		public const int DRAW_EVENT = 0x0002;
		public const int PREPROCESS_EVENT = 0x0003;

		[DllImport("gaussiansplatting", EntryPoint = "GetRenderEventFunc")] public static extern IntPtr GetRenderEventFunc();
		[DllImport("gaussiansplatting", EntryPoint = "IsAPIReady")] public static extern bool IsAPIReady();
		[DllImport("gaussiansplatting", EntryPoint = "GetLastMessage")] private static extern IntPtr _GetLastMessage();
		static public string GetLastMessage() { return Marshal.PtrToStringAnsi(_GetLastMessage()); }
		[DllImport("gaussiansplatting", EntryPoint = "LoadModel")] public static extern bool LoadModel(string file);
		[DllImport("gaussiansplatting", EntryPoint = "CopyModelToCuda")] public static extern int CopyModelToCuda();
		[DllImport("gaussiansplatting", EntryPoint = "RemoveModelFromCuda")] public static extern bool RemoveModelFromCuda(int model);
		[DllImport("gaussiansplatting", EntryPoint = "SetActiveModel")] public static extern void SetActiveModel(int model, bool active);
		[DllImport("gaussiansplatting", EntryPoint = "CreatePov")] public static extern int CreatePov();
		[DllImport("gaussiansplatting", EntryPoint = "RemovePov")] public static extern void RemovePov(int pov);
		[DllImport("gaussiansplatting", EntryPoint = "SetPovParameters")] public static extern void SetPovParameters(int pov, int width, int height);
		[DllImport("gaussiansplatting", EntryPoint = "IsInitialized")] public static extern bool IsInitialized(int pov);
		[DllImport("gaussiansplatting", EntryPoint = "GetTextureNativePointer")] public static extern IntPtr GetTextureNativePointer(int pov);
		[DllImport("gaussiansplatting", EntryPoint = "SetCameraDepthTextureNativePointer")] public static extern void SetCameraDepthTextureNativePointer(int pov, IntPtr ptr);
		[DllImport("gaussiansplatting", EntryPoint = "GetDepthTextureNativePointer")] public static extern IntPtr GetDepthTextureNativePointer(int pov);
		[DllImport("gaussiansplatting", EntryPoint = "SetDrawParameters")] public static extern void SetDrawParameters(int pov, int model, float[] position, float[] rotation, float[] scale, float[] proj, float fovy, float[] frustums);
		[DllImport("gaussiansplatting", EntryPoint = "SetModelCrop")] public static extern void SetModelCrop(int model, float[] box_min, float[] box_max);
		[DllImport("gaussiansplatting", EntryPoint = "GetModelCrop")] public static extern void GetModelCrop(int model, float[] box_min, float[] box_max);
		[DllImport("gaussiansplatting", EntryPoint = "IsDrawn")] public static extern bool IsDrawn(int pov);
		[DllImport("gaussiansplatting", EntryPoint = "IsPreprocessed")] public static extern bool IsPreprocessed(int pov);
		[DllImport("gaussiansplatting", EntryPoint = "GetNbSplat")] public static extern int GetNbSplat();
	}

	private class RegisteredModel
	{
		public GaussianSplattingModel model;
		public int modelId = 0;
		public bool isInError = false;
		public bool needToBeRemoved = false;
		public Bounds currentCropBox;
	}

	#region Members
	private State _state;
	private List<RegisteredModel> registeredModels = new List<RegisteredModel>();
	private System.Object loadModelMutex = new System.Object();
	protected HashSet<Observer> observers = new HashSet<Observer>();
	protected IntPtr renderEventFunc = IntPtr.Zero;
	protected Info info;
	protected int countDrawErrors;
	#endregion

	#region Getters / Setters
	public State state { get { return _state; }
		set {
			_state = value;
			foreach (Observer obs in observers)
            {
				obs.OnStateChanged(this, _state);
            }
		}
	}
	#endregion

	#region MonoBehaviour methods
	protected void Awake()
	{
		info = GetComponent<Info>();
	}

	protected void OnEnable()
	{
		StartCoroutine(Initialize());
	}

	protected void OnDisable()
	{
		countDrawErrors = 0;
		state = State.DISABLED;
	}

    private void OnDestroy()
    {
		state = State.PAUSE;

		observers.Clear();
        
		lock (registeredModels)
        {
			foreach (RegisteredModel m in registeredModels)
            {
				if (!m.isInError && m.modelId > 0)
                {
					Native.SetActiveModel(m.modelId, false);
					Native.RemoveModelFromCuda(m.modelId);
					m.modelId = 0;
				}
			}
			registeredModels.Clear();
		}
	}

    private void Update()
    {
        lock (registeredModels)
        {
            foreach (RegisteredModel m in registeredModels)
            {
                if (m.currentCropBox.size != Vector3.zero && m.currentCropBox != m.model.cropBox)
                {
                    float[] min = { m.model.cropBox.min.x, m.model.cropBox.min.y, m.model.cropBox.min.z };
                    float[] max = { m.model.cropBox.max.x, m.model.cropBox.max.y, m.model.cropBox.max.z };
                    Native.SetModelCrop(m.modelId, min, max);
                    m.currentCropBox = m.model.cropBox;
                }
            }

        }
    }

    #endregion

    #region Public methods
    public void AddObserver(Observer observer)
	{
		observers.Add(observer);
	}

	public void RemoveObserver(Observer observer)
	{
		if (!ReferenceEquals(this, observer)) observers.Remove(observer);
	}

	public void RegisterModel(GaussianSplattingModel model)
	{
		lock (registeredModels)
		{
			RegisteredModel registered_model = new RegisteredModel { model = model };
			registeredModels.Add(registered_model);
			Task.Run(async () => { await LoadTask(registered_model); });
		}
	}

	public void UnRegisterModel(GaussianSplattingModel model)
    {
		lock (registeredModels)
        {
			foreach (RegisteredModel m in registeredModels)
            {
				if (ReferenceEquals(model, m.model))
                {
					m.needToBeRemoved = true;
					m.model = null;
					break;
				}
            }
        }
	}

	public void PreProcessPass(int pov, Vector3 cam_pos, Quaternion cam_rot, Matrix4x4 proj_mat, float fovy)
	{
		if (state != State.RENDERING && state != State.PAUSE) return;

		lock (registeredModels)
        {
			int nb_active_models = 0;
			foreach (RegisteredModel m in registeredModels)
            {
				if (m.needToBeRemoved && !m.isInError && m.modelId > 0)
                {
					Native.SetActiveModel(m.modelId, false);
					Native.RemoveModelFromCuda(m.modelId);
					m.modelId = 0;
				}

				if (!m.isInError && m.modelId > 0)
                {
					bool active = m.model.gameObject.activeInHierarchy;
					Native.SetActiveModel(m.modelId, active);
								
					if (active)
                    {
						nb_active_models += 1;
						Vector3 pos = m.model.transform.InverseTransformPoint(cam_pos);
						Quaternion rot = Quaternion.Inverse(m.model.transform.rotation) * cam_rot;

						FrustumPlanes decomp = proj_mat.decomposeProjection;
						float[] position = { pos.x, pos.y, pos.z };
						float[] rotation = { rot.x, rot.y, rot.z, rot.w };
						float[] scale = { m.model.transform.lossyScale.x, m.model.transform.lossyScale.y, m.model.transform.lossyScale.z };
						float[] proj = matToFloat(proj_mat);
						float[] planes = { decomp.left, decomp.right, decomp.bottom, decomp.top, decomp.zNear, decomp.zFar };

						Native.SetDrawParameters(pov, m.modelId, position, rotation, scale, proj, fovy, planes);
					}
				}
			}

			if (nb_active_models == 0 && state == State.RENDERING)
            {
				state = State.PAUSE;
            }

			if (nb_active_models > 0 && state == State.PAUSE)
			{
				state = State.RENDERING;
			}
		}
	}

	public void SendPreprocessEvent()
    {
		if (state != State.RENDERING) return;
		lock (registeredModels) { if (registeredModels.Count == 0) return; }
		GL.IssuePluginEvent(renderEventFunc, Native.PREPROCESS_EVENT);
		GL.InvalidateState();
	}

	public void SendDrawEvent()
	{
		if (state != State.RENDERING) return;
		lock (registeredModels) { if (registeredModels.Count == 0) return; }
		GL.IssuePluginEvent(renderEventFunc, Native.DRAW_EVENT);
		GL.InvalidateState();
	}

	public bool WaitPovPreprocessed(int pov)
	{
		if (state != State.RENDERING) { return false; }
		lock (registeredModels) { if (registeredModels.Count == 0) return false; }

		float timestamp = Time.realtimeSinceStartup;
		bool ok = Native.IsPreprocessed(pov);
		while (!ok && Time.realtimeSinceStartup - timestamp < 1)
		{
			ok = Native.IsPreprocessed(pov);
		}
		
		if (!ok)
		{
			countDrawErrors += 1;
			if (countDrawErrors >= 5)
			{
				SetErrorState("Stop preprocessing error");
			}
		}
		else
		{
			countDrawErrors = 0;
		}

		return ok;
	}

	public bool WaitPovDrawn(int pov)
	{
		if (state != State.RENDERING) { return false; }
		lock (registeredModels) { if (registeredModels.Count == 0) return false; }

		float timestamp = Time.realtimeSinceStartup;
		bool ok = Native.IsDrawn(pov);
		while (!ok && Time.realtimeSinceStartup - timestamp < 1)
		{
			ok = Native.IsDrawn(pov);
		}

		if (!ok)
        {
			countDrawErrors += 1;
			if (countDrawErrors >= 5)
			{
				SetErrorState("Stop draw error");
			}
		}
		else
        {
			countDrawErrors = 0;
        }
		return ok;
	}
	#endregion

	#region Internal methods
	protected static float[] matToFloat(Matrix4x4 mat)
	{
		return new float[16]
		{
			mat.m00, mat.m10, mat.m20, mat.m30,
			mat.m01, mat.m11, mat.m21, mat.m31,
			mat.m02, mat.m12, mat.m22, mat.m32,
			mat.m03, mat.m13, mat.m23, mat.m33,
		};
	}

	protected void SetErrorState(string message, bool overload_gs_msg = false)
	{
		string gs_msg;

		if (overload_gs_msg)
		{
			gs_msg = message;

			Debug.LogError(message);
		}
		else
		{
			gs_msg = Native.GetLastMessage();

			Debug.LogError($"{message}: {gs_msg}");
		}

		if (info != null) info.lastMessage = gs_msg;

		state = State.ERROR;
	}

	private async Task LoadTask(RegisteredModel registeredModel)
	{
		try
		{
			await Task.Delay(1000);

			while (!Native.IsAPIReady()) await Task.Delay(10);

			//Avoid simultenaous loading
			lock (loadModelMutex)
            {
				//Model allready loaded
				if (!registeredModel.isInError && registeredModel.modelId > 0) return;

				string model_file_path = registeredModel.model.modelFilePath;
				if (string.IsNullOrEmpty(model_file_path)) { registeredModel.isInError = true; return; }

				if (!File.Exists(model_file_path))
				{
					registeredModel.isInError = true;
					return;
				}

				if (!Native.LoadModel(model_file_path))
				{
					registeredModel.isInError = true;
					return;
				}

				int modelid = Native.CopyModelToCuda();

				if (modelid <= 0)
				{
					registeredModel.isInError = true;
					return;
				}

				registeredModel.modelId = modelid;
                if (registeredModel.model.cropBox.size == Vector3.zero)
                {
                    float[] min = new float[3];
                    float[] max = new float[3];
                    Native.GetModelCrop(modelid, min, max);
                    registeredModel.model.cropBox.SetMinMax(new Vector3(min[0], min[1], min[2]), new Vector3(max[0], max[1], max[2]));
                }
                else
                {

                    float[] min = { registeredModel.model.cropBox.min.x, registeredModel.model.cropBox.min.y, registeredModel.model.cropBox.min.z };
                    float[] max = { registeredModel.model.cropBox.max.x, registeredModel.model.cropBox.max.y, registeredModel.model.cropBox.max.z };
                    Native.SetModelCrop(modelid, min, max);
                }
                registeredModel.currentCropBox = registeredModel.model.cropBox;
            }
        }
		catch (Exception ex)
		{
			Debug.LogException(ex);
		}
	}

	protected IEnumerator Initialize()
	{
		if (state != State.DISABLED && state != State.RENDERING) yield break;

		state = State.INITIALIZATION;

		while (!Native.IsAPIReady()) yield return new WaitForSecondsRealtime(0.01f);

		if (renderEventFunc == IntPtr.Zero)
		{
			renderEventFunc = Native.GetRenderEventFunc();
		}

		if (renderEventFunc == IntPtr.Zero)
		{
			SetErrorState("Cannot get Gaussian Splatting render event function.", true);
			yield break;
		}
		
		state = State.RENDERING;
	}
	#endregion

	#region POV interface
	public int CreatePOV(Vector2Int resolution, Texture2D depth)
	{
		int pov = Native.CreatePov();
		if (pov > 0)
		{
			Native.SetPovParameters(pov, resolution.x, resolution.y);
			Native.SetCameraDepthTextureNativePointer(pov, depth.GetNativeTexturePtr());
		}
		return pov;
	}
	
	public void RemovePOV(int pov)
    {
		Native.RemovePov(pov);
    }

	public void SendInitEvent()
    {
		GL.IssuePluginEvent(renderEventFunc, Native.INIT_EVENT);
	}

	public bool IsPovInitialized(int pov)
    {
		return Native.IsInitialized(pov);
    }

	public Texture2D CreateExternalPovTexture(int pov, Vector2Int resolution)
    {
		IntPtr ptr = Native.GetTextureNativePointer(pov);
		return Texture2D.CreateExternalTexture(resolution.x, resolution.y, TextureFormat.RGBAFloat, false, true, ptr);
	}

	public Texture2D CreateExternalPovDepthTexture(int pov, Vector2Int resolution)
	{
		IntPtr ptr = Native.GetDepthTextureNativePointer(pov);
		return Texture2D.CreateExternalTexture(resolution.x, resolution.y, TextureFormat.RFloat, false, true, ptr);
	}

	#endregion
}
