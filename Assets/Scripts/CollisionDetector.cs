using System.Collections;
using System.Collections.Generic;
using Unity.Sentis.Layers;
using UnityEngine;

public class CollisionDetector : MonoBehaviour
{
    Dictionary<string, bool> collisionTracker = new Dictionary<string, bool>();

    private void Start()
    {
        collisionTracker.Add("Ground", false);
        collisionTracker.Add("Reward", false);
    }

    void OnCollisionEnter(Collision col)
    {
        Dictionary<string, bool> newCollisions = new Dictionary<string, bool>();
        foreach (string tag in collisionTracker.Keys)
        {
            if (col.transform.CompareTag(tag))
                newCollisions[tag] = true;
            else
                newCollisions[tag] = collisionTracker[tag];
        }
        collisionTracker = newCollisions;
    }

    void OnCollisionExit(Collision col)
    {
        Dictionary<string, bool> newCollisions = new Dictionary<string, bool>();
        foreach (string tag in collisionTracker.Keys)
        {
            if (col.transform.CompareTag(tag))
                newCollisions[tag] = false;
            else
                newCollisions[tag] = collisionTracker[tag];
        }
        collisionTracker = newCollisions;
    }

    public bool Touching(string tag)
    {
        return collisionTracker[tag];
    }
}
