using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class SceneLoader : MonoBehaviour
{
    public List<int> scenes;

    void Start()
    {
        foreach (int scene in scenes) {
            SceneManager.LoadScene(scene, LoadSceneMode.Additive);
        }
    }
}
