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
        # We don't manage these with traitlets
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
    def handle_key(self, symbol, modifier):
        return False

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

    def handle_key(self, symbol, modifiers):
        if symbol == pyglet.window.key.A:
            self.pixel_size += 1
        elif symbol == pyglet.window.key.B:
            self.pixel_size = max(self.pixel_size - 1, 1)
        else:
            return False
        print(f"Pixel size is now {self.pixel_size}")
        return True

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
        super(CameraWatcher, self).__init__(*args, **kwargs)
        if not camera_pipe:
            camera_pipe = CameraPipe(width = self.width, height = self.height)#, cam_out = None)
        self.camera_pipe = camera_pipe
        pf = PixelizeEffect(pixel_size = 1)
        self.camera_pipe.effects.append(pf)
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
