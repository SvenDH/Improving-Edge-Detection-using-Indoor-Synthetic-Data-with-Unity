using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Windows;
using System.Linq;

public class CameraProcessing : MonoBehaviour
{
    [Serializable]
    public class CameraSettings
    {
        public enum textype
        {
            RGB,
            NORMAL,
            DEPTH,
            SEGMENTATION
        }

        public bool toggle = false;
        public Camera camera;
        public Shader shader;
        public RenderTexture texture;
        [Range(0, 5)]
        public float threshold;
        public RenderTextureFormat format;
        public RenderTextureReadWrite readwritemode;
        public textype type;
    }
    // 0: RGB
    // 1: SEGMENTATION + OUTLINE(alpha channel)
    // 2: DEPTH
    // 3: NORMALS

    public bool edgetoggle = false;

    public Shader alphaShader;
    private Material alphaMat;

    public List<CameraSettings> settings = new List<CameraSettings>();

    private void Start()
    {
        alphaMat = new Material(alphaShader);
        createTextures(Screen.width, Screen.height);
    }

    void Update()
    {
        if (Input.GetKeyDown("e"))
        {
            edgetoggle = !edgetoggle;
        }
        for (int i = 0; i < settings.Count; i++)
        {
            if (Input.GetKeyDown("" + i))
            {
                settings[i].toggle = !settings[i].toggle;
            }
        }
        if (settings.All(x => !x.toggle)) // If no toggle is on, toggle rgb on
        {
            settings[0].toggle = true;
        }
    }

    

    void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
        if (settings[0].texture != null)
        {
            Graphics.Blit(source, settings[(int)CameraSettings.textype.RGB].texture);

            RenderTexture.active = destination;
            GL.Clear(true, true, Color.black);

            if (edgetoggle)
            {
                Graphics.Blit(
                    settings[(int)CameraSettings.textype.SEGMENTATION].texture,
                    destination,
                    alphaMat);
            }
            else
            {
                for (int i = 0; i < settings.Count; i++)
                {
                    if (settings[i].toggle)
                    {
                        Graphics.Blit(
                            settings[i].texture,
                            destination);
                    }
                }
            }
        }
    }

    public void createTextures(int width, int height)
    {
        //Configure cameras
        for (int i = 0; i < settings.Count; i++)
        {
            settings[i].texture = new RenderTexture(width, height, 24, settings[i].format, settings[i].readwritemode);
            settings[i].texture.filterMode = FilterMode.Point;

            if (settings[i].camera != Camera.main)
                settings[i].camera.targetTexture = settings[i].texture;

            if (settings[i].shader)
                settings[i].camera.SetReplacementShader(settings[i].shader, "RenderType");
        }
    }

    public void cameraSettings(float FOV, float near, float far)
    {
        for (int i = 0; i < settings.Count; i++)
        {
            settings[i].camera.fieldOfView = FOV;
            settings[i].camera.nearClipPlane = near;
            settings[i].camera.farClipPlane = far;
        }
    }

}
