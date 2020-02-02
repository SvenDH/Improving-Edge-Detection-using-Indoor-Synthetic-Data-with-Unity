using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

public class ColorPicker : MonoBehaviour
{
    public bool object_class_color = true;

    // Start is called before the first frame update
    void Start() {
        if (object_class_color)
        {
            colorObjectsWithNames(new string[] { "Roof", "Bar", "House" },
                new Color32(0xff, 0x09, 0xe0, 0xff));
            colorObjectsWithNames(new string[] { "Wall" },
                new Color32(0x78, 0x78, 0x78, 0xff));
            colorObjectsWithNames(new string[] { "Ivy", "Tree", "Branch", "Plant" },
                new Color32(0x04, 0xc8, 0x03, 0xff));
            colorObjectsWithNames(new string[] { "Decor" },
                new Color32(0x00, 0xff, 0xcc, 0xff));
            colorObjectsWithNames(new string[] { "Floor" },
                new Color32(0x50, 0x32, 0x32, 0xff));
            colorObjectsWithNames(new string[] { "Ground" },
                new Color32(0x78, 0x78, 0x46, 0xff));
            colorObjectsWithNames(new string[] { "Cloth", "Shoe" },
                new Color32(0x00, 0x70, 0xff, 0xff));
            colorObjectsWithNames(new string[] { "Mug", "Bowl", "Can", "Kettle", "Dish" },
                new Color32(0x00, 0xff, 0x0a, 0xff));
            colorObjectsWithNames(new string[] { "Book" },
                new Color32(0xff, 0xa3, 0x00, 0xff));
            colorObjectsWithNames(new string[] { "Table", "Console" },
                new Color32(0xff, 0x06, 0x52, 0xff));
            colorObjectsWithNames(new string[] { "Desk" },
                new Color32(0x0a, 0xff, 0x47, 0xff));
            colorObjectsWithNames(new string[] { "Chair" },
                new Color32(0xcc, 0x46, 0x03, 0xff));
            colorObjectsWithNames(new string[] { "Armchair" },
                new Color32(0x08, 0xff, 0xd6, 0xff));
            colorObjectsWithNames(new string[] { "Bed" },
                new Color32(0xcc, 0x05, 0xff, 0xff));
            colorObjectsWithNames(new string[] { "Bookcase" },
                new Color32(0x00, 0xff, 0xf5, 0xff));
            colorObjectsWithNames(new string[] { "Commode", "Drawer", "Cupboard" },
                new Color32(0xe0, 0x05, 0xff, 0xff));
            colorObjectsWithNames(new string[] { "Closet", "Wardrobe" },
                new Color32(0x07, 0xff, 0xff, 0xff));
            colorObjectsWithNames(new string[] { "Displaycase" },
                new Color32(0x00, 0x00, 0xff, 0xff));
            colorObjectsWithNames(new string[] { "Couch", "Sofa" },
                new Color32(0x0b, 0x66, 0xff, 0xff));
            colorObjectsWithNames(new string[] { "Mirror" },
                new Color32(0xdc, 0xdc, 0xdc, 0xff));
            colorObjectsWithNames(new string[] { "Lamp", "Chandelier", "Sconce", "Light", "Spotlight" },
                new Color32(0xe0, 0xff, 0x08, 0xff));
            colorObjectsWithNames(new string[] { "Pissoir", "Toilet", "WC" },
                new Color32(0x00, 0xff, 0x85, 0xff));
            colorObjectsWithNames(new string[] { "Sink" },
                new Color32(0x00, 0xa3, 0xff, 0xff));
            colorObjectsWithNames(new string[] { "Candle" },
                new Color32(0xff, 0xad, 0x00, 0xff));
            colorObjectsWithNames(new string[] { "Picture" },
                new Color32(0xff, 0x06, 0x33, 0xff));
            colorObjectsWithNames(new string[] { "Frame" },
                new Color32(0x7f, 0x00, 0x00, 0xff));       //Custom
            colorObjectsWithNames(new string[] { "Vase" },
                new Color32(0x00, 0xff, 0xcc, 0xff));
            colorObjectsWithNames(new string[] { "Box" },
                new Color32(0x00, 0xff, 0x14, 0xff));
            colorObjectsWithNames(new string[] { "Basket", "Cart" },
                new Color32(0x5c, 0xff, 0x00, 0xff));
            colorObjectsWithNames(new string[] { "Carpet" },
                new Color32(0xff, 0x09, 0x5c, 0xff));
            colorObjectsWithNames(new string[] { "Trashcan" },
                new Color32(0xad, 0x00, 0xff, 0xff));
            colorObjectsWithNames(new string[] { "Footstool" },
                new Color32(0xff, 0x99, 0x00, 0xff));
            colorObjectsWithNames(new string[] { "Door" },
                new Color32(0x08, 0xff, 0x33, 0xff));
            colorObjectsWithNames(new string[] { "Stair" },
                new Color32(0x1f, 0x00, 0xff, 0xff));
            colorObjectsWithNames(new string[] { "Pillow" },
                new Color32(0x00, 0xeb, 0xff, 0xff));
            colorObjectsWithNames(new string[] { "Blanket", "Cover" },
                new Color32(0x14, 0x00, 0xff, 0xff));
            colorObjectsWithNames(new string[] { "Cable" },
                new Color32(0xff, 0xff, 0x3f, 0xff));       //Custom
            colorObjectsWithNames(new string[] { "Curtain" },
                new Color32(0xff, 0x33, 0x07, 0xff));
            colorObjectsWithNames(new string[] { "Bench" },
                new Color32(0xc2, 0xff, 0x00, 0xff));
            colorObjectsWithNames(new string[] { "Fireplace" },
                new Color32(0xfa, 0x0a, 0x0f, 0xff));
            colorObjectsWithNames(new string[] { "Fridge" },
                new Color32(0x14, 0xff, 0x00, 0xff));
            colorObjectsWithNames(new string[] { "Kitchen" },
                new Color32(0x00, 0xff, 0x29, 0xff));
            colorObjectsWithNames(new string[] { "Heater", "Radiator" },
                new Color32(0xff, 0xd6, 0x00, 0xff));
        }
        else
        {
            colorObjectsInstances();
        }


    }

    private void colorObjectsWithNames(string[] names, Color color)
    {
        foreach (Transform child in GetComponentsInChildren<Transform>()) {
            Renderer rend = child.gameObject.GetComponent<Renderer>();
            if (rend != null && names.Any(child.gameObject.name.Contains)) {   
                rend.material.SetColor("_SegColor", color);
            }
        }
    }

    static int objectinstance = 0;
    private void colorObjectsInstances()
    {
        foreach (Transform transform in transform)
        {
            foreach (Transform child in transform.GetComponentsInChildren<Transform>())
            {
                foreach (Transform gchild in child.GetComponentsInChildren<Transform>())
                {
                    Renderer rend = gchild.gameObject.GetComponent<Renderer>();
                    if (rend != null)
                    {
                        rend.material.SetColor("_SegColor", new Color32((byte)objectinstance, (byte)(objectinstance >> 8), (byte)(objectinstance >> 16), 0xff));
                    }
                }
                objectinstance++;
            }
        }
    }
}
