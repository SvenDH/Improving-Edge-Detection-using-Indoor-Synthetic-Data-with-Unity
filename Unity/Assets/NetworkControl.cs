using System;
using System.IO;
using System.Net;
using System.Threading;
using System.Net.Sockets;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityStandardAssets.Characters.FirstPerson;
using System.Linq;

public class NetworkControl : MonoBehaviour
{
    enum command
    {
        INIT = 0,
        START_RECORD,
        RECORD,
        DONE_RECORD,
        PLAYBACK,
        DONE_PLAYBACK,
        CAPTURE
    }

    enum state
    {
        START,
        INIT,
        PLAY,
        RECORD,
        PLAYBACK
    }

    public class SocketState
    {
        public Socket workSocket = null;
        public const int buffersize = 8142;
        public byte[] buffer = new byte[buffersize];
    }

    private state s = state.START;
    private int capturecount = 1;

    private int port = 49500;
    private Thread listenThread;
    private SocketState client = null;
    private static ManualResetEvent allDone = new ManualResetEvent(false);

    private Queue positionQueue = new Queue();

    private Camera camera;
    private CameraProcessing cameraProcessing;
    private Texture2D RGBACaptureTexture;
    private Texture2D RFloatCaptureTexture;

    private bool compress = true;
    private int width, height = -1;
    private float FOV, near, far;

    private void Start()
    {
        camera = Camera.main;
        cameraProcessing = camera.GetComponent<CameraProcessing>();
  
        if (Application.isBatchMode)
        {
            GetComponent<FirstPersonController>().enabled = false;
        }

        listenThread = new Thread(new ThreadStart(serverThreadFunc));
        listenThread.IsBackground = true;
        listenThread.Start();
    }

    void serverThreadFunc()
    {
        IPAddress ipAddress = IPAddress.Parse("127.0.0.1");
        IPEndPoint localEndPoint = new IPEndPoint(ipAddress, port);

        Socket listener = new Socket(ipAddress.AddressFamily, SocketType.Stream, ProtocolType.Tcp);

        try
        {
            listener.Bind(localEndPoint);
            listener.Listen(1);

            while (true)
            {
                allDone.Reset();
 
                Debug.Log("Waiting for a connection...");
                listener.BeginAccept(new AsyncCallback(AcceptCallback), listener);
                
                allDone.WaitOne();
            }
        }
        catch (Exception e)
        {
            Debug.Log(e.ToString());
        }
    }

    private void LateUpdate()
    {
        if (s == state.INIT)
        {
            if (Screen.width != width || Screen.height != height)
            {
                Screen.SetResolution(width, height, false);
            }
            else if (Screen.width == width && Screen.height == height)
            {
                //Send Camera info
                RGBACaptureTexture = new Texture2D(width, height, TextureFormat.RGBA32, false, false);
                RFloatCaptureTexture = new Texture2D(width, height, TextureFormat.RFloat, false, true);
                cameraProcessing.createTextures(width, height);
                cameraProcessing.cameraSettings(FOV, near, far);
                sendInfo();
                s = state.PLAY;
            }
        }

        if (Input.GetKeyDown("r"))
        {
            if (s == state.PLAY)
            {
                s = state.RECORD;
                SendData(command.START_RECORD, null, client);
            }
            else
            {
                s = state.PLAYBACK;
                SendData(command.DONE_RECORD, null, client);
            }
        }

        if (s == state.RECORD && Time.frameCount % capturecount == 0)
        {
            recordCharacter();
        }
        else if (s == state.PLAYBACK && positionQueue.Count > 0)
        {
            float[] pos = (float[])positionQueue.Dequeue(); // Dequeue new pose and capture
            moveCharacter(pos[1], pos[2], pos[3], pos[4], pos[5], pos[6]);

            List<byte[]> image_bytes = new List<byte[]>();
            image_bytes.Add(System.BitConverter.GetBytes((int)pos[0]));
            for (int i = 0; i < cameraProcessing.settings.Count; i++)
            {
                image_bytes.Add(saveRenderTexture(cameraProcessing.settings[i], compress));
            }
            byte[] data = Combine(image_bytes);
            SendData(command.CAPTURE, data, client);
        }

        if (!Application.isBatchMode)
        {
            GetComponent<FirstPersonController>().enabled = (s != state.PLAYBACK);
        }
    }

    private void moveCharacter(float x, float y, float z, float u, float v, float w)
    {
        // Move character to new pose
        transform.position = new Vector3(x, y, z);
        transform.localRotation = Quaternion.Euler(0, u * Mathf.Rad2Deg, 0);
        camera.transform.localRotation = Quaternion.Euler(v * Mathf.Rad2Deg, 0, w * Mathf.Rad2Deg);
    }

    private byte[] Combine(List<byte[]> arrays)
    {
        byte[] rv = new byte[arrays.Sum(a => a.Length)];
        int offset = 0;
        foreach (byte[] array in arrays)
        {
            System.Buffer.BlockCopy(array, 0, rv, offset, array.Length);
            offset += array.Length;
        }
        return rv;
    }

    private void sendInfo()
    {
        byte[] data = new byte[21];
        BitConverter.GetBytes(Screen.width).CopyTo(data, 0);
        BitConverter.GetBytes(Screen.height).CopyTo(data, 4);
        BitConverter.GetBytes(cameraProcessing.settings[0].camera.fieldOfView).CopyTo(data, 8);
        BitConverter.GetBytes(cameraProcessing.settings[0].camera.nearClipPlane).CopyTo(data, 12);
        BitConverter.GetBytes(cameraProcessing.settings[0].camera.farClipPlane).CopyTo(data, 16);
        BitConverter.GetBytes(compress).CopyTo(data, 20);

        SendData(command.INIT, data, client);
    }


    private void recordCharacter()
    {
        byte[] data = new byte[6 * 4];
        // Send position and rotation (in radiants)
        Vector3 pos = transform.position;
        BitConverter.GetBytes(pos.x).CopyTo(data, 0);
        BitConverter.GetBytes(pos.y).CopyTo(data, 4);
        BitConverter.GetBytes(pos.z).CopyTo(data, 8);
        BitConverter.GetBytes(transform.eulerAngles.y * Mathf.Deg2Rad).CopyTo(data, 12);
        BitConverter.GetBytes(camera.transform.eulerAngles.x * Mathf.Deg2Rad).CopyTo(data, 16);
        BitConverter.GetBytes(camera.transform.eulerAngles.z * Mathf.Deg2Rad).CopyTo(data, 20);

        SendData(command.RECORD, data, client);
    }


    private byte[] saveRenderTexture(CameraProcessing.CameraSettings settings, bool compress)
    {
        Texture2D captureTexture;

        RenderTexture.active = settings.texture;
        settings.camera.Render();

        if (settings.format == RenderTextureFormat.ARGB32)
        {
            captureTexture = RGBACaptureTexture;
        }
        else
        {
            captureTexture = RFloatCaptureTexture;
        }

        captureTexture.ReadPixels(new Rect(0, 0, Screen.width, Screen.height), 0, 0, false);
        captureTexture.Apply();

        byte[] data;
        byte[] img;
        if (compress)
        {
            img = captureTexture.EncodeToPNG();
            data = new byte[img.Length + 4];
            BitConverter.GetBytes(img.Length).CopyTo(data, 0);
            img.CopyTo(data, 4);
        }
        else
        {
            data = captureTexture.GetRawTextureData();
        }
        return data;
    }


    private void SendData(command c, byte[] data, SocketState client)
    {
        if (client != null)
        {
            byte[] bytes = new byte[(data != null ? data.Length : 0) + 8];
            BitConverter.GetBytes((data != null ? data.Length + 4 : 1)).CopyTo(bytes, 0);
            BitConverter.GetBytes((int)c).CopyTo(bytes, 4);
            if (data != null)
                data.CopyTo(bytes, 8);
            SocketAsyncEventArgs socketAsyncData = new SocketAsyncEventArgs();
            socketAsyncData.SetBuffer(bytes, 0, bytes.Length);
            client.workSocket.SendAsync(socketAsyncData);
        }
    }

    private void AcceptCallback(IAsyncResult ar)
    {
        allDone.Set();

        Socket listener = (Socket)ar.AsyncState;
        Socket handler = listener.EndAccept(ar);

        client = new SocketState();
        client.workSocket = handler;
        handler.BeginReceive(client.buffer, 0, SocketState.buffersize, 0,
            new AsyncCallback(ReceiveCallback), client);
        Debug.Log("Client connected");
    }

    private void ReceiveCallback(IAsyncResult AR)
    {
        SocketState sockstate = (SocketState)AR.AsyncState;
        Socket handler = sockstate.workSocket;

        int recieved = handler.EndReceive(AR);
        if (recieved <= 0)
            return;

        int processed = 0;
        while (processed < recieved)
        {
            int length = System.BitConverter.ToInt32(sockstate.buffer, processed);
            byte[] recData = new byte[length];
            Buffer.BlockCopy(sockstate.buffer, processed + 4, recData, 0, length);
            switch ((command)recData[0])
            {
                case command.INIT:
                    width = System.BitConverter.ToInt32(sockstate.buffer, 8);
                    height = System.BitConverter.ToInt32(sockstate.buffer, 12);
                    FOV = System.BitConverter.ToSingle(sockstate.buffer, 16);
                    near = System.BitConverter.ToSingle(sockstate.buffer, 20);
                    far = System.BitConverter.ToSingle(sockstate.buffer, 24);
                    compress = System.BitConverter.ToBoolean(sockstate.buffer, 28);
                    s = state.INIT;
                    break;
                case command.PLAYBACK:
                    positionQueue.Enqueue(new float[] {
                        System.BitConverter.ToSingle(recData, 4),
                        System.BitConverter.ToSingle(recData, 8),
                        System.BitConverter.ToSingle(recData, 12),
                        System.BitConverter.ToSingle(recData, 16),
                        System.BitConverter.ToSingle(recData, 20),
                        System.BitConverter.ToSingle(recData, 24),
                        System.BitConverter.ToSingle(recData, 28),
                    });
                    s = state.PLAYBACK;
                    break;
                case command.DONE_PLAYBACK:
                    s = state.PLAY;
                    break;
                default:
                    break;
            }
            processed += length + 4;
        }
        handler.BeginReceive(sockstate.buffer, 0, SocketState.buffersize, 0, new AsyncCallback(ReceiveCallback), sockstate);
    }

}