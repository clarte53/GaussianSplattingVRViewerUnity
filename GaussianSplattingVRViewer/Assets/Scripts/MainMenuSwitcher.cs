using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.UI;

public class MainMenuSwitcher : MonoBehaviour
{
    public InputActionReference menuActivate;
    public GameObject grabScaleWorldObject;
    public GameObject mainUiObject;
    public GameObject menuObject;
    public GameObject loadingObject;
    public float menuDeadZoneRotation = 45;
    public GaussianSplatting gs;
    public TMPro.TextMeshProUGUI fpsText;
    public TMPro.TextMeshProUGUI resolutionText;
    public TMPro.TextMeshProUGUI resolutionPercent;
    public Slider texScaleSlider;
    public TMPro.TextMeshProUGUI lastMessage;
    public TMPro.TextMeshProUGUI splatCount;
    public RawImage leftEye, rightEye;

    private System.Diagnostics.Stopwatch sw = System.Diagnostics.Stopwatch.StartNew();
    private int nb_frame = 0;
    private float cibleAngle = 0;

    void Start()
    {
        grabScaleWorldObject.SetActive(true);
        mainUiObject.SetActive(false);
        loadingObject.SetActive(false);
        menuObject.SetActive(false);
        menuActivate.action.performed += MenuActivate_performed;
        texScaleSlider.value = Mathf.Floor(gs.texFactor * 10);
        fpsText.text = "";
        resolutionText.text = "";
    }

    private void Update()
    {
        //Quit application on keyboard action
        if (Keyboard.current.qKey.wasReleasedThisFrame) { QuitApplication(); }
        if (Keyboard.current.escapeKey.wasReleasedThisFrame) { QuitApplication(); }

        //Show loading pannel
        if (!gs.loadModelEvent && !gs.loaded && !gs.isInError)
        {
            mainUiObject.SetActive(true);
            loadingObject.SetActive(true);
        }
        
        //Deactivate menu on the frame where data is loaded and not already initialized
        if (gs.loaded && !gs.initialized)
        {
            texScaleSlider.value = Mathf.Floor(gs.texFactor * 10);
            mainUiObject.SetActive(false);
            loadingObject.SetActive(false);
        }

        //Show menu to show error
        if (gs.isInError && !mainUiObject.activeSelf) {
            loadingObject.SetActive(false);
            ActivateMenu();
        }

        nb_frame += 1;
        if (sw.ElapsedMilliseconds > 250)
        {
            fpsText.text = string.Format("{0} FPS", (nb_frame * 1000) / sw.ElapsedMilliseconds);
            nb_frame = 0;
            sw.Restart();
        }
        resolutionText.text = gs.internalTexSize == Vector2.zero ? "" : string.Format("{0}x{1} px", gs.internalTexSize.x, gs.internalTexSize.y);
        resolutionPercent.text = string.Format("{0}%", Mathf.RoundToInt(gs.texFactor * 100));

        if (mainUiObject.activeSelf)
        {
            Vector3 cible = Camera.main.transform.position;
            mainUiObject.transform.position = Vector3.Lerp(mainUiObject.transform.position, cible, Time.deltaTime);

            Quaternion yangle = Quaternion.Euler(0, Camera.main.transform.rotation.eulerAngles.y, 0);
            Quaternion cangle = Quaternion.Euler(0, cibleAngle, 0);
            if (Quaternion.Angle(yangle, cangle) > menuDeadZoneRotation/2)
            {
                cibleAngle = Camera.main.transform.rotation.eulerAngles.y;
            }
            mainUiObject.transform.rotation = Quaternion.Lerp(mainUiObject.transform.rotation, Quaternion.Euler(0, cibleAngle, 0), Time.deltaTime);
        }
        lastMessage.text = gs.lastMessage;
        lastMessage.color = gs.isInError ? Color.red : Color.black;
        
        var nfi = (NumberFormatInfo)CultureInfo.InvariantCulture.NumberFormat.Clone();
        nfi.NumberGroupSeparator = " ";
        splatCount.text = gs.nb_splats.ToString("#,0", nfi);

        if (gs.tex.Length > 0 && gs.tex[0] != leftEye.texture)
        {
            leftEye.texture = gs.tex[0];
        }

        if (gs.tex.Length > 1 && gs.tex[1] != rightEye.texture)
        {
            rightEye.texture = gs.tex[1];
        }
    }

    private void MenuActivate_performed(InputAction.CallbackContext obj)
    {
        //if model is loading ignore event.
        if (!gs.loadModelEvent && !gs.loaded && !gs.isInError)
        {
            return;
        }
        
        if (grabScaleWorldObject.activeSelf)
        {
            ActivateMenu();
        } 
        else
        {
            CloseMenu();
        }
    }

    private void ActivateMenu()
    {
        grabScaleWorldObject.SetActive(false);
        mainUiObject.SetActive(true);
        menuObject.SetActive(true);
        mainUiObject.transform.position = Camera.main.transform.position;
        mainUiObject.transform.rotation = Quaternion.Euler(0, Camera.main.transform.rotation.eulerAngles.y, 0);
    }

    public void SliderValueChanged(float value)
    {
        gs.texFactor = value / 10;
    }

    public void QuitApplication()
    {
        Application.Quit();
    }

    public void CloseMenu()
    {
        grabScaleWorldObject.SetActive(true);
        mainUiObject.SetActive(false);
    }
}
