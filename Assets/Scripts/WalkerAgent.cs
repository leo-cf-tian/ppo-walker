using System;
using System.Collections;
using System.Collections.Generic;
using Unity.Burst.Intrinsics;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Unity.Sentis.Layers;
using Unity.VisualScripting;
using UnityEditor;
using UnityEngine;

public class WalkerAgent : Agent
{
    [Header("Body Parts")]
    public Transform hips;
    public Transform chest;
    public Transform spine;
    public Transform head;
    public Transform thighL;
    public Transform shinL;
    public Transform footL;
    public Transform thighR;
    public Transform shinR;
    public Transform footR;
    public Transform armL;
    public Transform forearmL;
    public Transform handL;
    public Transform armR;
    public Transform forearmR;
    public Transform handR;

    Dictionary<Transform, BodyPartController> bpDict = new Dictionary<Transform, BodyPartController>();

    public float positionSpring;
    public float positionDamper;
    public float maximumForce;

    // Start is called before the first frame update
    public override void Initialize()
    {
        CollectController(hips);
        CollectController(chest);
        CollectController(spine);
        CollectController(head);
        CollectController(thighL);
        CollectController(shinL);
        CollectController(footL);
        CollectController(thighR);
        CollectController(shinR);
        CollectController(footR);
        CollectController(armL);
        CollectController(forearmL);
        CollectController(handL);
        CollectController(armR);
        CollectController(forearmR);
        CollectController(handR);
    }
    public override void OnEpisodeBegin()
    {
        foreach (var bp in bpDict.Values)
        {
            bp.ResetPosition();
        }

        hips.rotation = Quaternion.Euler(0, UnityEngine.Random.Range(0.0f, 360.0f), 0);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        foreach (BodyPartController bp in bpDict.Values)
        {
            sensor.AddObservation(hips.InverseTransformDirection(bp.rb.velocity));
            sensor.AddObservation(hips.InverseTransformDirection(bp.rb.angularVelocity));

            sensor.AddObservation(hips.InverseTransformDirection(bp.rb.position - hips.position));

            bp.ObserveExertion(sensor);
        }
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        var i = -1;

        var continuousActions = actionBuffers.ContinuousActions;
        bpDict[chest].SetTargetRotation(continuousActions[++i], continuousActions[++i], continuousActions[++i]);
        bpDict[spine].SetTargetRotation(continuousActions[++i], continuousActions[++i], continuousActions[++i]);

        bpDict[thighL].SetTargetRotation(continuousActions[++i], continuousActions[++i], 0);
        bpDict[thighR].SetTargetRotation(continuousActions[++i], continuousActions[++i], 0);
        bpDict[shinL].SetTargetRotation(continuousActions[++i], 0, 0);
        bpDict[shinR].SetTargetRotation(continuousActions[++i], 0, 0);
        bpDict[footR].SetTargetRotation(continuousActions[++i], continuousActions[++i], continuousActions[++i]);
        bpDict[footL].SetTargetRotation(continuousActions[++i], continuousActions[++i], continuousActions[++i]);

        bpDict[armL].SetTargetRotation(continuousActions[++i], continuousActions[++i], 0);
        bpDict[armR].SetTargetRotation(continuousActions[++i], continuousActions[++i], 0);
        bpDict[forearmL].SetTargetRotation(continuousActions[++i], 0, 0);
        bpDict[forearmR].SetTargetRotation(continuousActions[++i], 0, 0);
        bpDict[head].SetTargetRotation(continuousActions[++i], continuousActions[++i], 0);

        //update joint strength settings
        bpDict[chest].SetStrength(continuousActions[++i]);
        bpDict[spine].SetStrength(continuousActions[++i]);
        bpDict[head].SetStrength(continuousActions[++i]);
        bpDict[thighL].SetStrength(continuousActions[++i]);
        bpDict[shinL].SetStrength(continuousActions[++i]);
        bpDict[footL].SetStrength(continuousActions[++i]);
        bpDict[thighR].SetStrength(continuousActions[++i]);
        bpDict[shinR].SetStrength(continuousActions[++i]);
        bpDict[footR].SetStrength(continuousActions[++i]);
        bpDict[armL].SetStrength(continuousActions[++i]);
        bpDict[forearmL].SetStrength(continuousActions[++i]);
        bpDict[armR].SetStrength(continuousActions[++i]);
        bpDict[forearmR].SetStrength(continuousActions[++i]);
    }

    void CollectController(Transform t)
    {
        BodyPartController bpController = t.gameObject.GetComponent<BodyPartController>();
        bpDict.Add(t, bpController);
        bpController.Initialize(new JointDrive {
            positionSpring = positionSpring,
            positionDamper = positionDamper,
            maximumForce = maximumForce
        });
    }

    void FixedUpdate()
    {
        if (hips.position.y < 1f)
            EndEpisode();

        AddReward(1);
    }
}
