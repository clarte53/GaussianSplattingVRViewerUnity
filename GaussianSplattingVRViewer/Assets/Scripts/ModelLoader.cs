using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.UI;
using UnityEngine.XR.Interaction.Toolkit;

public class ModelLoader : MonoBehaviour
{
    public TextAsset defaultModel;
    public GaussianSplattingModel modelPrefab;
    public ModelLoaderItem UIListItemPrefab;
    public RectTransform UIListElement;
    public UnityEvent NoModelLoaded;
    public MeasureLine measureLine;

    delegate void InstanciateModel(int idx, bool value);

    private string[] modelList;
    private bool[] modelLock;
    private GaussianSplattingModel[] models;

    void Start()
    {
        modelList = Directory.GetFiles(@".\", "*.ply");
        if (modelList.Length == 0)
        {
            string model_file_path = Application.temporaryCachePath + "/default.ply";
            File.WriteAllBytes(model_file_path, defaultModel.bytes);
            modelList = new string[] { model_file_path };
        }

        models = new GaussianSplattingModel[modelList.Length];
        modelLock = new bool[modelList.Length];

        for (int i = 0; i < modelList.Length; ++i)
        {
            modelLock[i] = false;
        }

        for (int i = 0; i < modelList.Length; ++i)
        {
            ModelLoaderItem listItem = Instantiate(UIListItemPrefab, UIListElement);
            listItem.index = i;
            listItem.loader = this;
            listItem.text.text = Path.GetFileName(modelList[i]);
        }
        
        NoModelLoaded.Invoke();
    }

    public void ToggleModel(int idx, bool value)
    {
        if (models[idx] == null)
        {
            models[idx] = Instantiate(modelPrefab);
            models[idx].modelFilePath = modelList[idx];
            models[idx].GetComponent<XRGrabInteractable>().selectEntered.AddListener(measureLine.select);
            models[idx].GetComponent<XRGrabInteractable>().selectExited.AddListener(measureLine.select);
        }

        //We are in the menu so we start deactivated
        models[idx].GetComponent<Collider>().enabled = false;
        models[idx].gameObject.SetActive(value);
    }

    public void LockModel(int idx, bool value)
    {
        modelLock[idx] = value;
        if (models[idx] != null && value)
        {
            models[idx].GetComponent<Collider>().enabled = false;
        }
    }

    public void DeactivateGrab()
    {
        foreach (GaussianSplattingModel model in models)
        {
            if (model != null)
            {
                //model.GetComponent<XRGrabInteractable>().enabled = false;
                model.GetComponent<Collider>().enabled = false;
            }
        }
    }

    public void ActivateGrab()
    {
        for (int i = 0; i < modelList.Length; ++i)
        {
            if (models[i] != null)
            {
                models[i].GetComponent<Collider>().enabled = true && !modelLock[i];
            }
        }
    }
}
