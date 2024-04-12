using System.Collections;
using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using UnityEngine;
using UnityEngine.Timeline;


public class BodyPartController : MonoBehaviour
{
    private bool hasActiveJoint;
    private ConfigurableJoint joint;
    [HideInInspector] public Rigidbody rb;

    WalkerAgent agent;

    private float normalizedCurrentTargetRotX = 0;
    private float normalizedCurrentTargetRotY = 0;
    private float normalizedCurrentTargetRotZ = 0;

    private float normalizedCurrentStrength = 0;

    private Vector3 initialPosition;
    private Quaternion initialRotation;

    public bool touchingGround = false;

    public void Initialize(WalkerAgent agent)
    {
        joint = GetComponent<ConfigurableJoint>();
        if (
            joint == null ||
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

        this.agent = agent;
    }

    public void ResetPosition()
    {
        rb.transform.position = initialPosition;
        rb.transform.rotation = initialRotation;
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
        float force = (s + 1f) * 0.5f * agent.maximumForce;

        var jd = new JointDrive
        {
            positionSpring = agent.positionSpring,
            positionDamper = agent.positionDamper,
            maximumForce = force
        };
        joint.slerpDrive = jd;

        normalizedCurrentStrength = force / agent.maximumForce;
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

    public float GetTorque()
    {
        if (hasActiveJoint)
            return joint.currentTorque.magnitude;
        return 0;
    }

    void OnCollisionEnter(Collision col)
    {
        if (col.transform.CompareTag("Ground"))
            touchingGround = true;
    }

    void OnCollisionExit(Collision other)
    {
        if (other.transform.CompareTag("Ground"))
            touchingGround = false;
    }
}
