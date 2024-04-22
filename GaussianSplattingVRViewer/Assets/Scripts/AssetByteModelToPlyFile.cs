using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class AssetByteModelToPlyFile : MonoBehaviour
{
    public GaussianSplattingModel gaussianModel;
    public TextAsset modelAsset;
    public string temporaryFileName;

    void Start()
    {
        string model_file_path = Application.temporaryCachePath + "/" + temporaryFileName;
        File.WriteAllBytes(model_file_path, modelAsset.bytes);
        gaussianModel.modelFilePath = model_file_path;
        gaussianModel.gameObject.SetActive(true);
    }
}
