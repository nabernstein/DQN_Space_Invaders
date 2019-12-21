from collections import deque
from skimage import transform
from skimage.color import rgb2gray
import numpy as np
from PIL import Image

class FrameStack:
    def __init__(self, maxlen):
        self.stack = deque(maxlen=maxlen)
        self.capacity = maxlen
        self.frame_no = 0
        self.save = False

    def add(self, observation):
        frame = self.preprocess_frame(observation)
        while len(self.stack) < self.capacity - 1:
            self.stack.append(frame)
        self.stack.append(frame)

    def reset(self, observation):
        self.stack.clear()
        self.add(observation)

    def get_state(self):
        return np.stack(self.stack, axis=2)

    def preprocess_frame(self, frame):
        gray = rgb2gray(frame)
        cropped_frame = gray[9:-14, 3:-15]
        # square_frame = transform.resize(cropped_frame, (cropped_frame.shape[0], cropped_frame.shape[0]))
        preprocessed_frame = np.array(transform.resize(cropped_frame, (84, 84)), dtype=np.float16)
        # preprocessed_frame = self.modify_grayscale(preprocessed_frame)

        if self.save:
            arr = np.expand_dims(np.array((preprocessed_frame * 255), dtype=np.uint8), axis=2)
            arrs = [arr for i in range(3)]
            stacked = np.stack(arrs, axis=2)
            stacked = stacked.reshape((84, 84, 3))
            img = Image.fromarray(stacked, 'RGB')
            img.save('./testimages/frame_%d.png' % self.frame_no)
            self.frame_no += 1

        return preprocessed_frame

    def save_frame(self, set_save):
        self.save = set_save

    def modify_grayscale(self, frame):
        for i in range(frame.shape[0]):
            for j in range(frame.shape[1]):
                frame[i][j] = 0 if frame[i][j] < 0.05 else 1.0
                # frame[i][j] *= 2.5
                # frame[i][j] = 1.0 if frame[i][j] > 1.0 else frame[i][j]

        return frame