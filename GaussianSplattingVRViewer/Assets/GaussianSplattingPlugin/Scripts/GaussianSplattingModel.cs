using System.Collections;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;

public class GaussianSplattingModel : MonoBehaviour
{
    public string modelFilePath = "";
    public Bounds cropBox;
    private GaussianSplatting gs;

    private void Awake()
    {
        gs = FindObjectOfType<GaussianSplatting>();
        gs.RegisterModel(this);
    }

    private void OnDestroy()
    {
        gs.UnRegisterModel(this);
    }
}
