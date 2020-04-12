import cv2
import numpy as np
import numba
import pyfakewebcam
import time
import pyglet # We use pyglet for displaying the images and handling events
import traitlets
import random

class CameraEffect(traitlets.HasTraits):
    pass

class CameraPipe(traitlets.HasTraits):
    input_device = traitlets.Unicode("/dev/video0")
    output_device = traitlets.Unicode("/dev/video1")
    width = traitlets.CInt(640)
    height = traitlets.CInt(480)
    fps = traitlets.CInt(30)
    effects = traitlets.List()
    cam_in = traitlets.Instance(cv2.VideoCapture)
    cam_out = traitlets.Instance(pyfakewebcam.FakeWebcam)

    def __init__(self, *args, **kwargs):
        super(CameraPipe, self).__init__(*args, **kwargs)
        # We don't manage these with traitlets
        self.output_arr1 = np.zeros((self.height, self.width, 3), dtype="u1")
        self.output_arr2 = np.zeros((self.height, self.width, 3), dtype="u1")

    @traitlets.default("cam_in")
    def _default_cam_in(self):
        return cv2.VideoCapture(self.input_device)

    @traitlets.default("cam_out")
    def _default_cam_out(self):
        return pyfakewebcam.FakeWebcam(self.output_device, self.width, self.height)

    def next_frame(self):
        _, self.output_arr1[:] = self.cam_in.read()
        #pixelize_frame(frame, output_arr)
        for effect in self.effects:
            effect(self.output_arr1, self.output_arr2)
            self.output_arr2, self.output_arr1 = self.output_arr1, self.output_arr2
        self.cam_out.schedule_frame(self.output_arr1)
        time.sleep(1/self.fps)

class PixelizeEffect(CameraEffect):
    pixel_size = traitlets.CInt(4)
    @traitlets.observe("pixel_size")
    def setup_function(self, change):
        pixel_size = self.pixel_size
        @numba.jit(nopython = True)
        def func(input_arr, output_arr):
            for i in range(input_arr.shape[0]):
                for j in range(input_arr.shape[1]):
                    for k in range(input_arr.shape[2]):
                        output_arr[i, j, 2 - k] = input_arr[(i // pixel_size) * pixel_size,
                                                            (j // pixel_size) * pixel_size,
                                                            k]

        self.func = func

    def __call__(self, input_arr, output_arr):
        return self.func(input_arr, output_arr)

@numba.jit(nopython=True)
def color_offset_frame(input_arr, output_arr):
    for i in range(input_arr.shape[0]):
        for j in range(input_arr.shape[1]):
            for k in range(input_arr.shape[2]):
                i1 = (i + pixel_size*k) % input_arr.shape[0]
                output_arr[i, j, 2 - k] = input_arr[i1, j, k]


class CameraWatcher(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        camera_pipe = kwargs.pop("camera_pipe", None)
        if not camera_pipe:
            camera_pipe = CameraPipe()
        self.camera_pipe = camera_pipe
        pf = PixelizeEffect(pixel_size = 1)
        self.camera_pipe.effects.append(pf)
        super(CameraWatcher, self).__init__(*args, **kwargs)

    def on_key_press(self, symbol, modifiers):
        ps = self.camera_pipe.effects[-1].pixel_size
        if symbol == pyglet.window.key.A:
            self.camera_pipe.effects[-1].pixel_size = max(ps + 1, 1)
        if symbol == pyglet.window.key.B:
            self.camera_pipe.effects[-1].pixel_size = max(ps - 1, 1)
        print("Pixel size", self.camera_pipe.effects[-1].pixel_size)

    def on_draw(self):
        self.camera_pipe.next_frame()

cw = CameraWatcher(width = 100, height = 100)

#cp = CameraPipe()
#pf = PixelizeEffect(pixel_size = 1)
#cp.effects.append(pf)
#i = 0
#while True:
    #cp.next_frame()
    #i += 1
    #if i % 100 == 0:
        #ps = random.randint(1, 10)
        #print(f"Changing pixel_size to {ps}")
        #pf.pixel_size = ps

pyglet.app.run()
