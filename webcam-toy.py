import cv2
import numpy as np
import numba
import pyfakewebcam
import time
import pyglet # We use pyglet for displaying the images and handling events
import traitlets
import ctypes

def get_numpy_data(arr):
     """
     :param arr: numpy array of float32
     :return: ctypes array of float32
     """
     # Accept any contiguous array of float32
     assert arr.flags["C_CONTIGUOUS"] or arr.flags["F_CONTIGUOUS"]
     assert arr.dtype == np.uint8
     return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8 * arr.size))[0]

class CameraPipe(traitlets.HasTraits):
    input_device = traitlets.Unicode("/dev/video0")
    output_device = traitlets.Unicode("/dev/video1")
    width = traitlets.CInt(640)
    height = traitlets.CInt(480)
    fps = traitlets.CInt(30)
    effects = traitlets.List()
    cam_in = traitlets.Instance(cv2.VideoCapture)
    cam_out = traitlets.Instance(pyfakewebcam.FakeWebcam, allow_none = True)

    def __init__(self, *args, **kwargs):
        super(CameraPipe, self).__init__(*args, **kwargs)
        # We don't manage these arrays with traitlets.
        # Pyglet does a lot of checking and byte-joining with arrays based on
        # pitch, so we hold on to things here and then read in an upside-down
        # way.
        self.display_array = np.zeros((self.height, self.width, 3), dtype="u1", order="C")
        self.output_arr1 = np.flipud(self.display_array)
        self.output_arr2 = np.zeros((self.height, self.width, 3), dtype="u1", order="C")

    @traitlets.default("cam_in")
    def _default_cam_in(self):
        in_cam = cv2.VideoCapture(self.input_device)
        in_cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        in_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        in_cam.set(cv2.CAP_PROP_FPS, self.fps)
        return in_cam

    @traitlets.default("cam_out")
    def _default_cam_out(self):
        return pyfakewebcam.FakeWebcam(self.output_device, self.width, self.height)

    def next_frame(self, dt = 0.0):
        _, self.output_arr1[:] = self.cam_in.read()
        #pixelize_frame(frame, output_arr)
        for effect in self.effects:
            effect(self.output_arr1, self.output_arr2)
            self.output_arr1[:] = self.output_arr2[:]
        if self.cam_out is not None:
            self.cam_out.schedule_frame(self.output_arr1)

class CameraEffect(traitlets.HasTraits):
    def __init__(self, *args, **kwargs):
        super(CameraEffect, self).__init__(*args, **kwargs)
        self.setup_function({})

    def handle_key(self, symbol, modifier):
        return False

    def __call__(self, input_arr, output_arr):
        return self.func(input_arr, output_arr)

class PixelizeEffect(CameraEffect):
    pixel_size = traitlets.CInt(4)
    @traitlets.observe("pixel_size")
    def setup_function(self, change):
        pixel_size = self.pixel_size
        # By the way, this *way* over JITs things -- need to turn pixel_size into an argument.
        @numba.jit(nopython = True)
        def func(input_arr, output_arr):
            for i in range(input_arr.shape[0]):
                for j in range(input_arr.shape[1]):
                    for k in range(input_arr.shape[2]):
                        output_arr[i, j, 2 - k] = input_arr[(i // pixel_size) * pixel_size,
                                                            (j // pixel_size) * pixel_size,
                                                            k]

        self.func = func

    def handle_key(self, symbol, modifiers):
        if symbol == pyglet.window.key.A:
            self.pixel_size += 1
        elif symbol == pyglet.window.key.B:
            self.pixel_size = max(self.pixel_size - 1, 1)
        else:
            return False
        print(f"Pixel size is now {self.pixel_size}")
        return True

class ColorOffsetEffect(CameraEffect):
    red_offset = traitlets.CInt(0)
    green_offset = traitlets.CInt(0)
    blue_offset = traitlets.CInt(0)

    @traitlets.observe("red_offset", "green_offset", "blue_offset")
    def setup_function(self, change):
        red_offset = self.red_offset
        blue_offset = self.blue_offset
        green_offset = self.green_offset
        @numba.jit(nopython=True)
        def func(input_arr, output_arr):
            for i in range(input_arr.shape[0]):
                for j in range(input_arr.shape[1]):
                    # red
                    i1 = (i + red_offset) % input_arr.shape[0]
                    output_arr[i, j, 0] = input_arr[i1, j, 0]
                    # green
                    i1 = (i + green_offset) % input_arr.shape[0]
                    output_arr[i, j, 1] = input_arr[i1, j, 1]
                    # blue
                    i1 = (i + blue_offset) % input_arr.shape[0]
                    output_arr[i, j, 2] = input_arr[i1, j, 2]
        self.func = func

    def handle_key(self, symbol, modifiers):
        sign = 1
        if modifiers & pyglet.window.key.MOD_SHIFT:
            sign = -1
        if symbol == pyglet.window.key.R:
            self.red_offset = max(0, sign + self.red_offset)
        elif symbol == pyglet.window.key.G:
            self.green_offset = max(0, sign + self.green_offset)
        elif symbol == pyglet.window.key.B:
            self.blue_offset = max(0, sign + self.blue_offset)
        else:
            return False
        return True

class CameraWatcher(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        camera_pipe = kwargs.pop("camera_pipe", None)
        super(CameraWatcher, self).__init__(*args, **kwargs)
        if not camera_pipe:
            camera_pipe = CameraPipe(width = self.width, height = self.height)#, cam_out = None)
        self.camera_pipe = camera_pipe
        self.camera_pipe.effects.append(PixelizeEffect(pixel_size = 1))
        self.camera_pipe.effects.append(ColorOffsetEffect(red_offset = 0, green_offset = 0, blue_offset = 0))
        self.image_data = pyglet.image.ImageData(
            self.camera_pipe.width, self.camera_pipe.height,
            'RGB', get_numpy_data(self.camera_pipe.display_array),
            1
        )
        self.camera_sprite = pyglet.sprite.Sprite(self.image_data, 0, 0)
        pyglet.clock.schedule_interval(self.camera_pipe.next_frame, 1.0/self.camera_pipe.fps)

    def on_key_press(self, symbol, modifiers):
        handled = False
        for effect in self.camera_pipe.effects:
            handled = handled or effect.handle_key(symbol, modifiers)
        if not handled:
            return super(CameraWatcher, self).on_key_press(symbol, modifiers)

    def update_image_data(self):
        self.image_data.set_data("RGB", 1, get_numpy_data(self.camera_pipe.display_array))

    def on_draw(self):
        self.clear()
        # pygarrayimage doesn't quite work anymore, and so we have to make lots of new arrays
        self.update_image_data()
        self.camera_sprite.image = self.image_data
        self.camera_sprite.draw()
        self.flip()

cw = CameraWatcher(width = 640, height = 480)
pyglet.app.run()
