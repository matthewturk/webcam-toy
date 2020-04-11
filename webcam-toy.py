import cv2
import numpy as np
import numba
import pyfakewebcam
import time

out_cam = pyfakewebcam.FakeWebcam('/dev/video1', 640, 480)
in_cam = cv2.VideoCapture("/dev/video0")
height, width = 640, 480
in_cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
in_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
in_cam.set(cv2.CAP_PROP_FPS, 30)

pixel_size = 8

@numba.jit(nopython=True)
def pixelize_frame(input_arr, output_arr):
    for i in range(input_arr.shape[0]):
        for j in range(input_arr.shape[1]):
            for k in range(input_arr.shape[2]):
                output_arr[i, j, 2 - k] = input_arr[(i // pixel_size) * pixel_size,
                                                    (j // pixel_size) * pixel_size,
                                                    k]

@numba.jit(nopython=True)
def color_offset_frame(input_arr, output_arr):
    for i in range(input_arr.shape[0]):
        for j in range(input_arr.shape[1]):
            for k in range(input_arr.shape[2]):
                i1 = (i + pixel_size*k) % input_arr.shape[0]
                output_arr[i, j, 2 - k] = input_arr[i1, j, k]

output_arr = np.zeros((480, 640, 3), dtype="u1")

N_frames = 300
frame_i = 0
t0 = time.time()
while True:
    if frame_i % N_frames == 0:
        fps = N_frames / (time.time() - t0)
        t0 = time.time()
        print(f"Rendering frame {frame_i} with FPS of {fps}")
    frame_i += 1
    _, frame = in_cam.read()
    #pixelize_frame(frame, output_arr)
    color_offset_frame(frame, output_arr)
    out_cam.schedule_frame(output_arr)
    #time.sleep(1/30.0)
