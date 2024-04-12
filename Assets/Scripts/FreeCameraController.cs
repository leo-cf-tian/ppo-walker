using System.Threading;
using UnityEngine;

public class FreeCameraController : MonoBehaviour
{
    [SerializeField] private float mouseSensitivity;
    [SerializeField] private float moveSpeed;
    [SerializeField] private bool holdToRotate = false;

    private float rotation = 0;
    private float xRotation = 180;

    private void Update()
    {
        MouseLook();
        Move();
    }

    private void MouseLook()
    {
        if (!holdToRotate || Input.GetMouseButton(0))
        {
            float mouseX = Input.GetAxis("Mouse X") * mouseSensitivity;
            float mouseY = Input.GetAxis("Mouse Y") * mouseSensitivity;

            rotation -= mouseY;
            rotation = Mathf.Clamp(rotation, -90f, 90f);

            xRotation += mouseX;

            transform.localRotation = Quaternion.Euler(rotation, 0f, 0f);
            transform.Rotate(Vector3.up * xRotation, Space.World);
        }
    }

    private void Move()
    {
        float verticalAxis = Input.GetKey(KeyCode.Space) ? (Input.GetKey(KeyCode.LeftShift) ? -1 : 1) : 0;
        float x = Input.GetAxis("Horizontal") * moveSpeed * Time.deltaTime;
        float z = Input.GetAxis("Vertical") * moveSpeed * Time.deltaTime;
        float y = verticalAxis * moveSpeed * Time.deltaTime;

        Vector3 movement = Quaternion.Euler(0, xRotation, 0f) * new Vector3(x, y, z);

        transform.Translate(movement, Space.World);
    }
}