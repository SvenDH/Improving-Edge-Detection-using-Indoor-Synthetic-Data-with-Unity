using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SwapObjects : MonoBehaviour
{
    private List<GameObject> objects = new List<GameObject>();
    void Start()
    {
        foreach (Transform child in GetComponentsInChildren<Transform>())
        {
            Renderer rend = child.gameObject.GetComponent<Renderer>();
            if (rend != null)
            {
                objects.Add(child.gameObject);
            }
        }
    }

    void Update()
    {
        foreach (GameObject obj in objects)
        {
            int rand = Random.Range(0, 3);
            obj.SetActive(rand == 0 ? false : true);
        }
    }

    public static float NextGaussian()
    {
        float v1, v2, s;
        do
        {
            v1 = 2.0f * Random.Range(0f, 1f) - 1.0f;
            v2 = 2.0f * Random.Range(0f, 1f) - 1.0f;
            s = v1 * v1 + v2 * v2;
        } while (s >= 1.0f || s == 0f);

        s = Mathf.Sqrt((-2.0f * Mathf.Log(s)) / s);

        return v1 * s;
    }

    public static float NextGaussian(float mean, float standard_deviation)
    {
        return mean + NextGaussian() * standard_deviation;
    }
}
