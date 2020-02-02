using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SobelOutline : MonoBehaviour
{
    public Shader outline;
    public Shader alphaPack;
    private Material outlineMat;
    private Material packMat;

    [Range(0, 3)]
    public float SobelResolution = 1;

    private CameraProcessing cameraProcessing;

    void Start()
    {
        outlineMat = new Material(outline);
        packMat = new Material(alphaPack);
        cameraProcessing = Camera.main.GetComponent<CameraProcessing>();
    }

    void OnPostRender()
    {
        RenderTexture sobelTexture = RenderTexture.GetTemporary(Screen.width, Screen.height, 24, RenderTextureFormat.ARGB32);

        packMat.SetTexture("_AlphaTex", sobelTexture);
        packMat.SetFloat("_ResX", Screen.width * SobelResolution);
        packMat.SetFloat("_ResY", Screen.height * SobelResolution);
        outlineMat.SetTexture("_DepthTex", cameraProcessing.settings[(int)CameraProcessing.CameraSettings.textype.DEPTH].texture);
        outlineMat.SetFloat("_ResX", Screen.width * SobelResolution);
        outlineMat.SetFloat("_ResY", Screen.height * SobelResolution);

        for (int i = 0; i < cameraProcessing.settings.Count; i++)
        {
            Graphics.Blit(
                cameraProcessing.settings[i].texture,
                sobelTexture,
                outlineMat);

            packMat.SetFloat("_Threshold", cameraProcessing.settings[i].threshold);
            Graphics.Blit(
                cameraProcessing.settings[(int)CameraProcessing.CameraSettings.textype.SEGMENTATION].texture,
                cameraProcessing.settings[(int)CameraProcessing.CameraSettings.textype.SEGMENTATION].texture,
                packMat);
        }

        RenderTexture.ReleaseTemporary(sobelTexture);
    }
}
