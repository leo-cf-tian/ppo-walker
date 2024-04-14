using System.Collections;
using System.Collections.Generic;
using Unity.Sentis.Layers;
using UnityEngine;

public class CollisionDetector : MonoBehaviour
{
    Dictionary<string, bool> collissionTracker = new Dictionary<string, bool>();

    private void Start()
    {
        collissionTracker.Add("Ground", false);
    }

    void OnCollisionEnter(Collision col)
    {
        foreach (string tag in collissionTracker.Keys)
        {
            if (col.transform.CompareTag(tag))
            {
                collissionTracker[tag] = true;
            }
        }
    }

    void OnCollisionExit(Collision col)
    {
        foreach (string tag in collissionTracker.Keys)
        {
            if (col.transform.CompareTag(tag))
            {
                collissionTracker[tag] = false;
            }
        }
    }

    public bool Touching(string tag)
    {
        return collissionTracker[tag];
    }
}
