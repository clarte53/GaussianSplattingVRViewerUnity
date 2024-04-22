using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RenderCallbacksTest : MonoBehaviour
{
	protected void OnPreRender()
	{
		Debug.Log("["+Time.frameCount + "] RenderCallbacksTest::OnPreRender() {" + name + "}");
	}

	[ImageEffectOpaque]
	protected void OnRenderImage(RenderTexture source, RenderTexture destination)
	{
		Debug.Log("[" + Time.frameCount + "] RenderCallbacksTest::OnRenderImage() {" + name + "}");
		Graphics.Blit(source, destination);
	}
}
