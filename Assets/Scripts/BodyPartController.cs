using System.Collections;
using System.Collections.Generic;
using Unity.MLAgents.Sensors;
using UnityEngine;
using UnityEngine.Timeline;

public class BodyPartController : MonoBehaviour
{
    private bool hasActiveJoint;
    private ConfigurableJoint joint;
    [HideInInspector] public Rigidbody rb;

    JointDrive maxJd;

    private float normalizedCurrentTargetRotX;
    private float normalizedCurrentTargetRotY;
    private float normalizedCurrentTargetRotZ;

    private float normalizedCurrentStrength;

    private Vector3 initialPosition;
    private Quaternion initialRotation;

    public void Initialize(JointDrive maxJd)
    {
        joint = GetComponent<ConfigurableJoint>();
        if (joint == null ||
            joint.angularXMotion == ConfigurableJointMotion.Locked &&
            joint.angularYMotion == ConfigurableJointMotion.Locked &&
            joint.angularZMotion == ConfigurableJointMotion.Locked
        )
            hasActiveJoint = false;
        else
            hasActiveJoint = true;

        rb = GetComponent<Rigidbody>();

        initialPosition = rb.position;
        initialRotation = rb.rotation;

        this.maxJd = maxJd;
    }

    public void ResetPosition()
    {
        rb.position = initialPosition;
        rb.rotation = initialRotation;
        rb.velocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;
    }

    public void SetTargetRotation(float x, float y, float z)
    {
        x = (x + 1f) * 0.5f;
        z = (y + 1f) * 0.5f;
        z = (z + 1f) * 0.5f;

        float xRot = Mathf.Lerp(joint.lowAngularXLimit.limit, joint.highAngularXLimit.limit, x);
        float yRot = Mathf.Lerp(-joint.angularYLimit.limit, joint.angularYLimit.limit, y);
        float zRot = Mathf.Lerp(-joint.angularZLimit.limit, joint.angularZLimit.limit, z);

        joint.targetRotation = Quaternion.Euler(xRot, yRot, zRot);

        normalizedCurrentTargetRotX = Mathf.Clamp(x, 0, 1);
        normalizedCurrentTargetRotX = Mathf.Clamp(y, 0, 1);
        normalizedCurrentTargetRotX = Mathf.Clamp(z, 0, 1);
    }

    public void SetStrength(float s)
    {
        float force = (s + 1f) * 0.5f * maxJd.maximumForce;

        var jd = new JointDrive
        {
            positionSpring = maxJd.positionSpring,
            positionDamper = maxJd.positionDamper,
            maximumForce = force
        };
        joint.slerpDrive = jd;

        normalizedCurrentStrength = force / maxJd.maximumForce;
    }

    public void ObserveExertion(VectorSensor sensor)
    {
        if (hasActiveJoint)
        {
            if (joint.angularXMotion != ConfigurableJointMotion.Locked)
                sensor.AddObservation(normalizedCurrentTargetRotX);

            if (joint.angularYMotion != ConfigurableJointMotion.Locked)
                sensor.AddObservation(normalizedCurrentTargetRotY);

            if (joint.angularYMotion != ConfigurableJointMotion.Locked)
                sensor.AddObservation(normalizedCurrentTargetRotZ);

            sensor.AddObservation(normalizedCurrentStrength);
        }
    }
}
