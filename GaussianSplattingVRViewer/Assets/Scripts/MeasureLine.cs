using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR.Interaction.Toolkit;

public class MeasureLine : MonoBehaviour
{
    public Transform left;
    public Transform right;
    public LineRenderer line;
    public TMPro.TextMeshPro text;
    int nbSelected = 0;

    void Awake()
    {
        if (line == null) { line = GetComponentInChildren<LineRenderer>(true); }
        if (text == null) { text = GetComponentInChildren<TMPro.TextMeshPro>(true); }
        if (line != null) { line.gameObject.SetActive(false); }
        if (text != null) { text.gameObject.SetActive(false); }
    }

    void Update()
    {
        if (line != null)
        {
            line.SetPosition(0, left.position);
            line.SetPosition(1, right.position);
        }
        if (text != null)
        {
            float v = Vector3.Distance(left.position, right.position);
            if (v < 1)
            {
                text.text = string.Format("{0} cm", Mathf.RoundToInt(100 * v));
            }
            else
            {
                text.text = string.Format("{0:0.00} m", v);
            }
            text.transform.position = Vector3.Lerp(left.position, right.position, 0.5f) + Camera.main.transform.forward * -0.01f;
            text.transform.rotation = Camera.main.transform.rotation;
        }
    }

    public void select(SelectEnterEventArgs _)
    {
        nbSelected += 1;
        if (nbSelected == 2)
        {
            if (line != null) { line.gameObject.SetActive(true); }
            if (text != null) { text.gameObject.SetActive(true); }
        }
    }

    public void select(SelectExitEventArgs _)
    {
        nbSelected -= 1;
        if (line != null) { line.gameObject.SetActive(false); }
        if (text != null) { text.gameObject.SetActive(false); }
    }
}
