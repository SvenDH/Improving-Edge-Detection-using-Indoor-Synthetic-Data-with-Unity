using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Lights : MonoBehaviour
{
    private List<Light> lights = new List<Light>();
    void Start()
    {
        foreach (Transform child in GetComponentsInChildren<Transform>())
        {
            Light light = child.gameObject.GetComponent<Light>();
            if (light != null)
            {
                lights.Add(light);
            }
        }
    }

    void Update()
    {
        foreach (Light light in lights)
        {
            int rand = Random.Range(0, 2);
            light.gameObject.SetActive(rand == 0 ? true : false);
        }
    }
}
