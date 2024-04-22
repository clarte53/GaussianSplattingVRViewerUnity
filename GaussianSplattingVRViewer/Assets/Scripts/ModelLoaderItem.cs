using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class ModelLoaderItem : MonoBehaviour
{
    public int index;
    public ModelLoader loader;
    public Text text;

    public void ToggleModel(bool value)
    {
        loader.ToggleModel(index, value);
    }

    public void LockModel(bool value)
    {
        loader.LockModel(index, value);
    }
}
