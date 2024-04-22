using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR;

[RequireComponent(typeof(Camera))]
public class GaussianSplattingCamera : MonoBehaviour
{
	[Header("Init Parameters")]
	public bool isXr = false;
	public Material renderMaterial;
	public Material depthMaterial;
	
	[Header("Dynamic Parameters")]
	[Range(0.1f, 1f)]
	public float texFactor = 0.7f;
	
	private GaussianSplatting gs = null;
	private Camera cam;
	private int[] pov = { 0, 0 };
	private Coroutine initCo = null;

	protected Texture2D[] tex;
	protected Texture2D[] depthTex;
	protected Texture2D[] camDepthTex;
	protected RenderTexture[] camDepthRTex;
	protected GameObject real_leye, real_reye;
	protected float lastTexFactor = 0f;

	public Vector2Int InternalTexSize { get; private set; } = Vector2Int.zero;

	#region MonoBehaviour methods
	private void OnEnable()
	{
		cam = GetComponent<Camera>();
		gs = FindObjectOfType<GaussianSplatting>();
		initCo = StartCoroutine(Initialize());
	}

	private void OnDisable()
	{
		if (initCo != null)
        {
			StopCoroutine(initCo);
			initCo = null;
		}
		RemovePOV();
	}

	private void Update()
    {
        if (texFactor != lastTexFactor && initCo == null)
        {
			lastTexFactor = texFactor;
			initCo = StartCoroutine(Initialize());
		}
    }

    protected void OnPreRender()
	{
		//Do nothing until initialization is done.
		if (pov[0] <= 0 || renderMaterial == null || (gs.state != GaussianSplatting.State.RENDERING && gs.state != GaussianSplatting.State.PAUSE)) { return; }

		bool doit = true;

		if (isXr)
		{
			if (TryGetEyesPoses(out Vector3 lpos, out Vector3 rpos, out Quaternion lrot, out Quaternion rrot))
			{
				if (real_leye == null) { real_leye = new GameObject("real leye"); real_leye.transform.parent = cam.transform.parent; }
				real_leye.transform.localPosition = lpos;
				real_leye.transform.localRotation = lrot;

				if (real_reye == null) { real_reye = new GameObject("real reye"); real_reye.transform.parent = cam.transform.parent; }
				real_reye.transform.localPosition = rpos;
				real_reye.transform.localRotation = rrot;
			}
			else
			{
				doit = false;
			}
		}

		if (doit)
		{
			for (int i = 0; i < (isXr ? 2 : 1); ++i)
			{
				if (tex != null && tex[i] != null)
				{
					float fovy = cam.fieldOfView * Mathf.PI / 180;
					Matrix4x4 proj_mat = cam.projectionMatrix;
					Vector3 cam_pos = cam.transform.position;
					Quaternion cam_rot = cam.transform.rotation;

					if (isXr)
					{
						if (i == 0)
						{
							proj_mat = cam.GetStereoProjectionMatrix(Camera.StereoscopicEye.Left);
							cam_pos = real_leye.transform.position;
							cam_rot = real_leye.transform.rotation;
						}
						else
						{
							proj_mat = cam.GetStereoProjectionMatrix(Camera.StereoscopicEye.Right);
							cam_pos = real_reye.transform.position;
							cam_rot = real_reye.transform.rotation;
						}

					}

					gs.PreProcessPass(pov[i], cam_pos, cam_rot, proj_mat, fovy);
				}
			}

			gs.SendPreprocessEvent();
		}
	}

	[ImageEffectOpaque]
	protected void OnRenderImage(RenderTexture source, RenderTexture destination)
	{
		//Do nothing until initialization is done.
		if (pov[0] <= 0 || gs.state != GaussianSplatting.State.RENDERING || renderMaterial == null) {
			Graphics.Blit(source, destination);
			return;
		}

		//Grab depth texture
		if (depthMaterial != null)
		{
			for (int i = 0; i < (isXr ? 2 : 1); ++i)
			{
				depthMaterial.SetFloat("_Scale", 1.0f);
				depthMaterial.SetInt("_Eye", i);

				Graphics.Blit(source, camDepthRTex[i], depthMaterial);
				Graphics.CopyTexture(camDepthRTex[i], camDepthTex[i]);
			}
		}

		if (!gs.WaitPovPreprocessed(pov[0]))
        {
			Graphics.Blit(source, destination);
			return;
		}

		if (isXr && !gs.WaitPovPreprocessed(pov[1]))
		{
			Graphics.Blit(source, destination);
			return;
		}

		gs.SendDrawEvent();

		if (!gs.WaitPovDrawn(pov[0]))
		{
			Graphics.Blit(source, destination);
			return;
		}

		if (isXr && !gs.WaitPovDrawn(pov[1]))
		{
			Graphics.Blit(source, destination);
			return;
		}

		renderMaterial.SetFloat("_Scale", 1.0f);
		for (int i = 0; i < (isXr ? 2 : 1); ++i)
        {
			renderMaterial.SetTexture(i == 0 ? "_GaussianSplattingTexLeftEye" : "_GaussianSplattingTexRightEye", tex[i]);
			renderMaterial.SetTexture(i == 0 ? "_GaussianSplattingDepthTexLeftEye" : "_GaussianSplattingDepthTexRightEye", depthTex[i]);
		}
		Graphics.Blit(source, destination, renderMaterial);
	}
	#endregion

    #region Internal methods

	protected void RemovePOV()
    {
		if (pov[0] > 0)
		{
			int pov0 = pov[0];
			pov[0] = 0;

			if (isXr)
			{
				gs.RemovePOV(pov[1]);
			}
			gs.RemovePOV(pov0);

		}
	}

	protected IEnumerator Initialize()
    {
		if (initCo != null) yield break;

		yield return new WaitUntil(() => { return gs.state == GaussianSplatting.State.RENDERING; });

		//If pov is already created, remove it
		RemovePOV();

		tex = new Texture2D[isXr ? 2 : 1];
		depthTex = new Texture2D[isXr ? 2 : 1];
		camDepthTex = new Texture2D[isXr ? 2 : 1];
		camDepthRTex = new RenderTexture[isXr ? 2 : 1];
		lastTexFactor = texFactor;

		//If isXr Wait for XR to be ready. When stereoActiveEye is not Mono that is the camera is ready with correct pixelWidth & pixelHeight.
		if (isXr)
		{
			yield return new WaitUntil(() => {
				return cam.stereoActiveEye != Camera.MonoOrStereoscopicEye.Mono;
			});
		}

		Vector2Int resolution = new Vector2Int((int)(cam.pixelWidth * texFactor), (int)(cam.pixelHeight * texFactor));
		InternalTexSize = resolution;
		int[] temp_pov = { 0, 0 };
		for (int i = 0; i < (isXr ? 2 : 1); ++i)
		{
			camDepthRTex[i] = new RenderTexture(resolution.x, resolution.y, 0, RenderTextureFormat.RFloat, 0);
			camDepthRTex[i].enableRandomWrite = true;
			camDepthRTex[i].Create();

			camDepthTex[i] = new Texture2D(resolution.x, resolution.y, TextureFormat.RFloat, false);
			temp_pov[i] = gs.CreatePOV(resolution, camDepthTex[i]);
		}

		gs.SendInitEvent();
		
		for (int i = 0; i < (isXr ? 2 : 1); ++i)
        {

			yield return new WaitUntil(() => gs.IsPovInitialized(temp_pov[i]));
			
			tex[i] = gs.CreateExternalPovTexture(temp_pov[i], resolution);
			depthTex[i] = gs.CreateExternalPovDepthTexture(temp_pov[i], resolution);
		}

		//Set pov value when all is ready
		if (isXr)
        {
			pov[1] = temp_pov[1];
		}
		pov[0] = temp_pov[0];
		initCo = null;
	}

	protected static bool TryGetEyesPoses(out Vector3 lpos, out Vector3 rpos, out Quaternion lrot, out Quaternion rrot)
	{
		List<XRNodeState> states = new List<XRNodeState>();
		InputTracking.GetNodeStates(states);

		lpos = Vector3.zero;
		rpos = Vector3.zero;
		lrot = Quaternion.identity;
		rrot = Quaternion.identity;

		int nbfound = 0;

		foreach (XRNodeState state in states)
		{
			if (state.tracked && state.nodeType == XRNode.LeftEye)
			{
				if (state.TryGetPosition(out Vector3 tpos)) { lpos = tpos; nbfound += 1; }
				if (state.TryGetRotation(out Quaternion trot)) { lrot = trot; nbfound += 1; }
			}
			if (state.tracked && state.nodeType == XRNode.RightEye)
			{
				if (state.TryGetPosition(out Vector3 tpos)) { rpos = tpos; nbfound += 1; }
				if (state.TryGetRotation(out Quaternion trot)) { rrot = trot; nbfound += 1; }
			}
		}

		return nbfound == 4;
	}
	#endregion
}
