using UnityEngine;

[RequireComponent(typeof(GaussianSplatting))]
public class TrackTRS : MonoBehaviour
{
	#region Members
	public Transform tracked;

	protected GaussianSplatting gs;
	#endregion

	#region MonoBehaviour methods
	protected void Awake()
	{
		gs = GetComponent<GaussianSplatting>();
	}

	protected void Update()
	{
		if (tracked != null)
		{
			gs.transform.localScale = tracked.localScale;
			transform.localPosition = tracked.localPosition;
			transform.localRotation = tracked.localRotation;
		}
	}
	#endregion
}
