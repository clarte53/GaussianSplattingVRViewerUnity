using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(BoxCollider))]
public class SelectionBox : MonoBehaviour
{
    public GaussianSplattingModel gsModel;
    public Material lineMaterial;
    public Color lineColor;
    public Color lineSelectedColor;
    [Range(0.001f, 1f)]
    public float lineWidth = 0.1f;

    BoxCollider box = null;
    LineRenderer[] lines = null;
    Color currentColor;

    void Start()
    {
        box = GetComponent<BoxCollider>();
        currentColor = lineColor;
        lines = new LineRenderer[6];
        for (int i = 0; i < 6; ++i)
        {
            GameObject child = new GameObject();
            child.transform.parent = transform;
            child.transform.localPosition = Vector3.zero;
            child.transform.localRotation = Quaternion.identity;
            lines[i] = child.AddComponent<LineRenderer>();
            lines[i].positionCount = 4;
            lines[i].loop = true;
            lines[i].useWorldSpace = false;
            lines[i].gameObject.SetActive(false);
        }
    }

    void Update()
    {
        box.center = gsModel.cropBox.center;
        box.size = gsModel.cropBox.size;
        UpdateLines();
    }

    void UpdateLines()
    {
        Vector3 min = gsModel.cropBox.min;
        Vector3 max = gsModel.cropBox.max;

        for (int i = 0; i < 6; ++i)
        {
            lines[i].startWidth = lineWidth * transform.lossyScale.x;
            lines[i].endWidth = lineWidth * transform.lossyScale.x;
            lines[i].material = lineMaterial;
            lines[i].material.color = currentColor;
            lines[i].startColor = currentColor;
            lines[i].endColor = currentColor;
        }

        lines[0].SetPositions(new Vector3[] { new Vector3(min.x, min.y, min.z), new Vector3(max.x, min.y, min.z), new Vector3(max.x, max.y, min.z), new Vector3(min.x, max.y, min.z) });
        lines[1].SetPositions(new Vector3[] { new Vector3(min.x, min.y, max.z), new Vector3(max.x, min.y, max.z), new Vector3(max.x, max.y, max.z), new Vector3(min.x, max.y, max.z) });
        lines[2].SetPositions(new Vector3[] { new Vector3(min.x, min.y, min.z), new Vector3(max.x, min.y, min.z), new Vector3(max.x, min.y, max.z), new Vector3(min.x, min.y, max.z) });
        lines[3].SetPositions(new Vector3[] { new Vector3(min.x, max.y, min.z), new Vector3(max.x, max.y, min.z), new Vector3(max.x, max.y, max.z), new Vector3(min.x, max.y, max.z) });
        lines[4].SetPositions(new Vector3[] { new Vector3(min.x, min.y, min.z), new Vector3(min.x, max.y, min.z), new Vector3(min.x, max.y, max.z), new Vector3(min.x, min.y, max.z) });
        lines[5].SetPositions(new Vector3[] { new Vector3(max.x, min.y, min.z), new Vector3(max.x, max.y, min.z), new Vector3(max.x, max.y, max.z), new Vector3(max.x, min.y, max.z) });
    }

    public void ActivateLines(bool value)
    {
        if (lines != null && lines.Length == 6)
        {
            for (int i = 0; i < 6; ++i)
            {
                lines[i].gameObject.SetActive(value);
            }
        }
    }

    public void SelectLines(bool value)
    {
        currentColor = value ? lineSelectedColor : lineColor;
    }
}
