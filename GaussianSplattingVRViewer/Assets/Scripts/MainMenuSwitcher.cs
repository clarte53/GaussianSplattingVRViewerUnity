using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.InputSystem;
using UnityEngine.UI;

public class MainMenuSwitcher : MonoBehaviour, GaussianSplatting.Observer
{
    public InputActionReference menuActivate;
    public GameObject mainUiObject;
    public GameObject menuObject;
    public GameObject loadingObject;
    public float menuDeadZoneRotation = 45;
    public GaussianSplattingCamera gscam;
    public Info gsInfo;
    public TMPro.TextMeshProUGUI fpsText;
    public TMPro.TextMeshProUGUI resolutionText;
    public TMPro.TextMeshProUGUI resolutionPercent;
    public Slider texScaleSlider;
    public TMPro.TextMeshProUGUI lastMessage;
    public TMPro.TextMeshProUGUI splatCount;
    public UnityEvent OnOpenMainMenu;
    public UnityEvent OnCloseMainMenu;

    private System.Diagnostics.Stopwatch sw = System.Diagnostics.Stopwatch.StartNew();
    private int nb_frame = 0;
    private float cibleAngle = 0;

    private void Awake()
    {
        GaussianSplatting gs = FindObjectOfType<GaussianSplatting>();
        gs?.AddObserver(this);

        mainUiObject.SetActive(false);
        loadingObject.SetActive(false);
        menuObject.SetActive(true);
        menuActivate.action.performed += MenuActivate_performed;
        texScaleSlider.value = Mathf.Floor(gscam.texFactor * 10);
        fpsText.text = "";
        resolutionText.text = "";
    }

    private void OnDestroy()
    {
        GaussianSplatting gs = FindObjectOfType<GaussianSplatting>();
        gs?.RemoveObserver(this);
    }

    private void Update()
    {
        //Quit application on keyboard action
        if (Keyboard.current.qKey.wasReleasedThisFrame) { QuitApplication(); }
        if (Keyboard.current.escapeKey.wasReleasedThisFrame) { QuitApplication(); }

        nb_frame += 1;
        if (sw.ElapsedMilliseconds > 250)
        {
            fpsText.text = string.Format("{0} FPS", (nb_frame * 1000) / sw.ElapsedMilliseconds);
            nb_frame = 0;
            sw.Restart();
        }
        resolutionText.text = gscam.InternalTexSize == Vector2.zero ? "" : string.Format("{0}x{1} px", gscam.InternalTexSize.x, gscam.InternalTexSize.y);
        resolutionPercent.text = string.Format("{0}%", Mathf.RoundToInt(gscam.texFactor * 100));

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
        lastMessage.text = gsInfo.lastMessage;
        //lastMessage.color = gsInfo.isInError ? Color.red : Color.black;
        
        var nfi = (NumberFormatInfo)CultureInfo.InvariantCulture.NumberFormat.Clone();
        nfi.NumberGroupSeparator = " ";
        splatCount.text = gsInfo.nb_splats.ToString("#,0", nfi);
    }

    private void MenuActivate_performed(InputAction.CallbackContext obj)
    {
        if (mainUiObject.activeSelf)
        {
            CloseMenu();
        } 
        else
        {
            ActivateMenu();
        }
    }

    public void ActivateMenu()
    {
        mainUiObject.SetActive(true);
        menuObject.SetActive(true);
        mainUiObject.transform.position = Camera.main.transform.position;
        mainUiObject.transform.rotation = Quaternion.Euler(0, Camera.main.transform.rotation.eulerAngles.y, 0);
        OnOpenMainMenu.Invoke();
    }

    public void SliderValueChanged(float value)
    {
        gscam.texFactor = value / 10;
    }

    public void QuitApplication()
    {
        Application.Quit();
    }

    public void CloseMenu()
    {
        mainUiObject.SetActive(false);
        OnCloseMainMenu.Invoke();
    }

    public void OnStateChanged(GaussianSplatting gs, GaussianSplatting.State state)
    {
        if (state == GaussianSplatting.State.ERROR)
        {
            loadingObject.SetActive(false);
            ActivateMenu();
        }
    }
}
