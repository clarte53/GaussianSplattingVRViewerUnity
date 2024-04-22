using UnityEngine;

[RequireComponent(typeof(GaussianSplatting))]
public class Info : MonoBehaviour, GaussianSplatting.Observer
{
	#region Members
	public string lastMessage = "";
	public int nb_splats = 0;

	protected GaussianSplatting gs;
	#endregion

	#region MonoBehaviour methods
	protected void Start()
	{
		gs = GetComponent<GaussianSplatting>();
		
		gs.AddObserver(this);
	}

	protected void Update()
	{
		if (gs.state >= GaussianSplatting.State.DISABLED)
		{
			lastMessage = GaussianSplatting.Native.GetLastMessage();
			nb_splats = GaussianSplatting.Native.GetNbSplat();
		}
	}
	#endregion

	#region Observer implementation
	public void OnStateChanged(GaussianSplatting gs, GaussianSplatting.State state)
	{
		switch (state)
		{
			case GaussianSplatting.State.DISABLED:
				lastMessage = "";
				nb_splats = 0;

				break;
			default:
				break;
		}
	}
	#endregion
}
