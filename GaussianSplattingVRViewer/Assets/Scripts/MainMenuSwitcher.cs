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
    public GameObject menuObject;
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

        //Show menu to show error
        if (gs.isInError && !menuObject.activeSelf) { ActivateMenu(); }

        nb_frame += 1;
        if (sw.ElapsedMilliseconds > 250)
        {
            fpsText.text = string.Format("{0} FPS", (nb_frame * 1000) / sw.ElapsedMilliseconds);
            nb_frame = 0;
            sw.Restart();
        }
        resolutionText.text = gs.internalTexSize == Vector2.zero ? "" : string.Format("{0}x{1} px", gs.internalTexSize.x, gs.internalTexSize.y);
        resolutionPercent.text = string.Format("{0}%", Mathf.RoundToInt(gs.texFactor * 100));

        if (menuObject.activeSelf)
        {
            Vector3 cible = Camera.main.transform.position;
            menuObject.transform.position = Vector3.Lerp(menuObject.transform.position, cible, Time.deltaTime);

            Quaternion yangle = Quaternion.Euler(0, Camera.main.transform.rotation.eulerAngles.y, 0);
            Quaternion cangle = Quaternion.Euler(0, cibleAngle, 0);
            if (Quaternion.Angle(yangle, cangle) > menuDeadZoneRotation/2)
            {
                cibleAngle = Camera.main.transform.rotation.eulerAngles.y;
            }
            menuObject.transform.rotation = Quaternion.Lerp(menuObject.transform.rotation, Quaternion.Euler(0, cibleAngle, 0), Time.deltaTime);
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
        if (grabScaleWorldObject.activeSelf)
        {
            ActivateMenu();
        } 
        else
        {
            grabScaleWorldObject.SetActive(true);
            menuObject.SetActive(false);
        }
    }

    private void ActivateMenu()
    {
        grabScaleWorldObject.SetActive(false);
        menuObject.SetActive(true);
        menuObject.transform.position = Camera.main.transform.position;
        menuObject.transform.rotation = Quaternion.Euler(0, Camera.main.transform.rotation.eulerAngles.y, 0);
    }

    public void SliderValueChanged(float value)
    {
        gs.texFactor = value / 10;
    }

    public void QuitApplication()
    {
        Application.Quit();
    }
}
