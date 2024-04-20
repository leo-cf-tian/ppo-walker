using System;
using System.Collections;
using System.Collections.Generic;
using Unity.Burst.Intrinsics;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Unity.Sentis;
using Unity.Sentis.Layers;
using Unity.VisualScripting;
using UnityEditor;
using UnityEngine;
using UnityEngine.UIElements.Experimental;
using UnityEngine.XR;

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

    [Header("Orientation Box")]
    public Transform orientationBox;
    public Transform rotationIndicator;

    [Header("Reward")]
    public Transform rewardBall;
    private Vector3 initialRewardPosition;

    [Header("Movement Variables")]
    public float positionSpring;
    public float positionDamper;
    public float maximumForce;

    public float targetSpeed;

    Dictionary<Transform, BodyPartController> bpDict = new Dictionary<Transform, BodyPartController>();
    RayPerceptionSensorComponent3D raySensor;

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

        raySensor = head.GetComponent<RayPerceptionSensorComponent3D>();

        initialRewardPosition = rewardBall.transform.position;
    }
    public override void OnEpisodeBegin()
    {
        foreach (var bp in bpDict.Values)
            bp.ResetPosition();

        Quaternion rot = Quaternion.Euler(0, UnityEngine.Random.Range(0.0f, 360.0f), 0);
        hips.rotation = rot;
        orientationBox.rotation = rot;

        Vector3 rand = rot * new Vector3(0, UnityEngine.Random.Range(-0.5f, 0.5f), UnityEngine.Random.Range(10f, 20f));
        rand = Quaternion.Euler(0, UnityEngine.Random.Range(-45f, 45f), 0) * rand;
        rewardBall.position = initialRewardPosition + rand;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        foreach (BodyPartController bp in bpDict.Values)
        {
            sensor.AddObservation(orientationBox.InverseTransformDirection(bp.rb.velocity));
            sensor.AddObservation(orientationBox.InverseTransformDirection(bp.rb.angularVelocity));

            sensor.AddObservation(orientationBox.InverseTransformDirection(bp.rb.position - hips.position));
            sensor.AddObservation(bp.GetTorque());

            sensor.AddObservation(bp.collisions.Touching("Ground"));
            sensor.AddObservation(bp.collisions.Touching("Reward"));

            bp.ObserveExertion(sensor);
        }

        sensor.AddObservation(orientationBox.rotation.eulerAngles.y);
        sensor.AddObservation(Quaternion.FromToRotation(hips.forward, orientationBox.forward));
        sensor.AddObservation(Quaternion.FromToRotation(head.forward, orientationBox.forward));

        sensor.AddObservation(GetWeightedVelocity());
        sensor.AddObservation(targetSpeed);
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        var i = -1;

        var continuousActions = actionBuffers.ContinuousActions;

        // Update target rotations
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

        // Update joint strengths
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

        // Update orientation
        orientationBox.Rotate(0, continuousActions[++i] * Time.fixedDeltaTime * 5 * 180, 0);
        rotationIndicator.rotation = orientationBox.rotation;
    }

    Vector3 GetWeightedVelocity()
    {
        Vector3 momentum = Vector3.zero;
        float totalMass = 0;

        foreach (BodyPartController bp in bpDict.Values)
        {
            momentum += bp.rb.velocity * bp.rb.mass;
            totalMass += bp.rb.mass;
        }

        return momentum / totalMass;
    }

    void CollectController(Transform t)
    {
        BodyPartController bpController = t.gameObject.GetComponent<BodyPartController>();
        bpDict.Add(t, bpController);
        bpController.Initialize(this);
    }

    void FixedUpdate()
    {
        orientationBox.position = new Vector3(hips.position.x, rotationIndicator.position.y, hips.position.z);
        rotationIndicator.position = new Vector3(orientationBox.position.x, rotationIndicator.position.y, orientationBox.position.z);

        foreach (BodyPartController bp in bpDict.Values)
        {
            if (bp != bpDict[footL] && bp != bpDict[footR] && bp.collisions.Touching("Ground"))
            {
                AddReward(-500);
                EndEpisode();
                return;
            }

            if (bp.collisions.Touching("Reward"))
            {
                AddReward(500);
                RandomizeReward();
            }
        }

        foreach (BodyPartController bp in bpDict.Values)
            AddReward(-bp.GetTorque() / 10000);

        if (CanSeeReward())
            AddReward(0.5f);

        Vector3 diff = rewardBall.position - orientationBox.position;
        float facing = Vector2.Dot(new Vector2(orientationBox.forward.x, orientationBox.forward.z).normalized, new Vector2(diff.x, diff.z).normalized);
        AddReward((facing - 0.5f) * 5);

        diff = orientationBox.position - head.position;
        facing = Vector2.Dot(new Vector2(head.forward.x, head.forward.z).normalized, new Vector2(diff.x, diff.z).normalized);
        AddReward((facing - 0.5f) * 2);

        diff = orientationBox.position - hips.position;
        facing = Vector2.Dot(new Vector2(hips.forward.x, hips.forward.z).normalized, new Vector2(diff.x, diff.z).normalized);
        AddReward((facing - 0.5f) * 2);

        facing = Vector3.Dot(GetWeightedVelocity().normalized, new Vector3(orientationBox.forward.x, 0, orientationBox.forward.z).normalized);
            AddReward(Mathf.Pow(GetWeightedVelocity().magnitude, 2) * (facing - 0.7f) * 2 - Mathf.Abs(hips.forward.y));

        if (bpDict[footL].collisions.Touching("Ground") ^ bpDict[footR].collisions.Touching("Ground"))
            AddReward(2);

        AddReward(3f);
    }

    bool CanSeeReward()
    {
        // from https://forum.unity.com/threads/how-to-check-ray-perception-sensor-3d.1420445/
        var rayOutputs = RayPerceptionSensor.Perceive(raySensor.GetRayPerceptionInput(), false).RayOutputs;
        int lengthOfRayOutputs = rayOutputs.Length;

        for (int i = 0; i < lengthOfRayOutputs; i++)
        {
            GameObject goHit = rayOutputs[i].HitGameObject;
            if (goHit != null && goHit.tag == "Reward")
                return true;
        }

        return false;
    }

    void RandomizeReward()
    {
        Vector3 rand;
        do
        {
            rand = new Vector3(UnityEngine.Random.Range(-10f, 10f), UnityEngine.Random.Range(-0.5f, 0.5f), UnityEngine.Random.Range(-10f, 10f));
            rand += initialRewardPosition;
        }
        while ((rand - hips.position).magnitude < 4);
        rewardBall.position = rand;
    }
}
