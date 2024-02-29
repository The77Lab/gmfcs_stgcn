import numpy as np
import torch
import functools
from typing import Callable, Type, Union
import random

class DataAugmenter:

    def __init__(self, 
                 noise = True, 
                 shift = True, 
                 flip = True,
                 noise_scale = 0.03,
                 max_shift = 0.03,
                 ):
        self.noise = noise
        self.noise_scale = noise_scale
        self.shift = shift
        self.max_shift = max_shift
        self.flip = flip
    
    def __call__(self, data):
        # print(data.shape)
        if self.noise == True:
            noise = np.random.normal(scale=self.noise_scale, size=data.shape[2:])
            new_shape = (1, 1, ) + noise.shape
            broadcasted_noise = np.broadcast_to(noise.reshape(new_shape), data.shape)
            data = data + broadcasted_noise
            data = np.clip(data, -1, 1)

        if self.shift == True:
            shift = np.random.uniform(-self.max_shift, self.max_shift, size=(1,))
            shift_tiled = np.tile(shift, reps=(2, data.shape[1], data.shape[2]))
            data = data + shift_tiled

        data = np.clip(data, -1, 1)
        if self.flip == True:
            flip_chance = random.random()
            if flip_chance >= 0.5:
                data = -data
        return data


class PreNormalize2D:
    """Normalize the range of keypoint values. """

    def __init__(self, img_shape=(1080, 1920), threshold=0.01, mode='auto', data_augmentation=False):
        self.threshold = threshold
        # Will skip points with score less than threshold
        self.img_shape = img_shape
        self.mode = mode
        self.x_data_augmenter = DataAugmenter(flip=True)
        self.y_data_augmenter = DataAugmenter(flip=False)
        self.data_augmentation = data_augmentation
        assert mode in ['fix', 'auto']

    def __call__(self, results):
        mask, maskout, keypoint = None, None, results['keypoint'].astype(np.float32)
        if 'keypoint_score' in results:
            keypoint_score = results.pop('keypoint_score').astype(np.float32)
            keypoint = np.concatenate([keypoint, keypoint_score[..., None]], axis=-1)

        if keypoint.shape[-1] == 3:
            mask = keypoint[..., 2] > self.threshold
            maskout = keypoint[..., 2] <= self.threshold

        if self.mode == 'auto':
            if mask is not None:
                if np.sum(mask):
                    x_max, x_min = np.max(keypoint[mask, 0]), np.min(keypoint[mask, 0])
                    y_max, y_min = np.max(keypoint[mask, 1]), np.min(keypoint[mask, 1])
                else:
                    x_max, x_min, y_max, y_min = 0, 0, 0, 0
            else:
                x_max, x_min = np.max(keypoint[..., 0]), np.min(keypoint[..., 0])
                y_max, y_min = np.max(keypoint[..., 1]), np.min(keypoint[..., 1])
            if (x_max - x_min) > 10 and (y_max - y_min) > 10:
                keypoint[..., 0] = (keypoint[..., 0] - (x_max + x_min) / 2) / (x_max - x_min) * 2
                keypoint[..., 1] = (keypoint[..., 1] - (y_max + y_min) / 2) / (y_max - y_min) * 2
        else:
            h, w = results.get('img_shape', self.img_shape)
            keypoint[..., 0] = (keypoint[..., 0] - (w / 2)) / (w / 2)
            keypoint[..., 1] = (keypoint[..., 1] - (h / 2)) / (h / 2)
        
        if self.data_augmentation:
            keypoint[..., 0] = self.x_data_augmenter(keypoint[..., 0])
            keypoint[..., 1] = self.y_data_augmenter(keypoint[..., 1])

        if maskout is not None:
            keypoint[..., 0][maskout] = 0
            keypoint[..., 1][maskout] = 0
        results['keypoint'] = keypoint

        return results


    


class JointToBone:

    def __init__(self, dataset='nturgb+d', target='keypoint'):
        self.dataset = dataset
        self.target = target
        if self.dataset not in ['nturgb+d', 'openpose', 'coco', 'handmp']:
            raise ValueError(
                f'The dataset type {self.dataset} is not supported')
        if self.dataset == 'nturgb+d':
            self.pairs = ((0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5), (7, 6), (8, 20), (9, 8),
                          (10, 9), (11, 10), (12, 0), (13, 12), (14, 13), (15, 14), (16, 0), (17, 16), (18, 17),
                          (19, 18), (21, 22), (20, 20), (22, 7), (23, 24), (24, 11))
        elif self.dataset == 'openpose':
            self.pairs = ((0, 0), (1, 0), (2, 1), (3, 2), (4, 3), (5, 1), (6, 5), (7, 6), (8, 2), (9, 8), (10, 9),
                          (11, 5), (12, 11), (13, 12), (14, 0), (15, 0), (16, 14), (17, 15))
        elif self.dataset == 'coco':
            self.pairs = ((0, 0), (1, 0), (2, 0), (3, 1), (4, 2), (5, 0), (6, 0), (7, 5), (8, 6), (9, 7), (10, 8),
                          (11, 0), (12, 0), (13, 11), (14, 12), (15, 13), (16, 14))
        elif self.dataset == 'handmp':
            self.pairs = ((0, 0), (1, 0), (2, 1), (3, 2), (4, 3), (5, 0), (6, 5), (7, 6), (8, 7), (9, 0), (10, 9),
                          (11, 10), (12, 11), (13, 0), (14, 13), (15, 14), (16, 15), (17, 0), (18, 17), (19, 18),
                          (20, 19))

    def __call__(self, results):

        keypoint = results['keypoint']
        M, T, V, C = keypoint.shape
        bone = np.zeros((M, T, V, C), dtype=np.float32)

        assert C in [2, 3]
        for v1, v2 in self.pairs:
            bone[..., v1, :] = keypoint[..., v1, :] - keypoint[..., v2, :]
            if C == 3 and self.dataset in ['openpose', 'coco', 'handmp']:
                score = (keypoint[..., v1, 2] + keypoint[..., v2, 2]) / 2
                bone[..., v1, 2] = score

        results[self.target] = bone
        return results
class ToMotion:

    def __init__(self, dataset='nturgb+d', source='keypoint', target='motion'):
        self.dataset = dataset
        self.source = source
        self.target = target

    def __call__(self, results):
        data = results[self.source]
        M, T, V, C = data.shape
        motion = np.zeros_like(data)

        assert C in [2, 3]
        motion[:, :T - 1] = np.diff(data, axis=1)
        if C == 3 and self.dataset in ['openpose', 'coco']:
            score = (data[:, :T - 1, :, 2] + data[:, 1:, :, 2]) / 2
            motion[:, :T - 1, :, 2] = score

        results[self.target] = motion

        return results
class MergeSkeFeat:
    def __init__(self, feat_list=['keypoint'], target='keypoint', axis=-1):
        """Merge different feats (ndarray) by concatenate them in the last axis. """

        self.feat_list = feat_list
        self.target = target
        self.axis = axis

    def __call__(self, results):
        feats = []
        for name in self.feat_list:
            feats.append(results.pop(name))
        feats = np.concatenate(feats, axis=self.axis)
        results[self.target] = feats
        return results
class Rename:
    """Rename the key in results.

    Args:
        mapping (dict): The keys in results that need to be renamed. The key of
            the dict is the original name, while the value is the new name. If
            the original name not found in results, do nothing.
            Default: dict().
    """

    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, results):
        for key, value in self.mapping.items():
            if key in results:
                assert isinstance(key, str) and isinstance(value, str)
                assert value not in results, ('the new name already exists in '
                                              'results')
                results[value] = results[key]
                results.pop(key)
        return results
class GenSkeFeat:
    def __init__(self, dataset='nturgb+d', feats=['j'], axis=-1):
        self.dataset = dataset
        self.feats = feats
        self.axis = axis
        ops = []
        if 'b' in feats or 'bm' in feats:
            ops.append(JointToBone(dataset=dataset, target='b'))
        ops.append(Rename({'keypoint': 'j'}))
        if 'jm' in feats:
            ops.append(ToMotion(dataset=dataset, source='j', target='jm'))
        if 'bm' in feats:
            ops.append(ToMotion(dataset=dataset, source='b', target='bm'))
        ops.append(MergeSkeFeat(feat_list=feats, axis=axis))
        # self.ops = Compose(ops)
        self.ops = ops
        

    def __call__(self, results):
        if 'keypoint_score' in results and 'keypoint' in results:
            assert self.dataset != 'nturgb+d'
            assert results['keypoint'].shape[-1] == 2, 'Only 2D keypoints have keypoint_score. '
            keypoint = results.pop('keypoint')
            keypoint_score = results.pop('keypoint_score')
            results['keypoint'] = np.concatenate([keypoint, keypoint_score[..., None]], -1)
        for i in self.ops:
            results = i(results)
        return results
        # return self.ops(results)


class UniformSampleFrames:
    """Uniformly sample frames from the video.

    To sample an n-frame clip from the video. UniformSampleFrames basically
    divide the video into n segments of equal length and randomly sample one
    frame from each segment. To make the testing results reproducible, a
    random seed is set during testing, to make the sampling results
    deterministic.

    Required keys are "total_frames", "start_index" , added or modified keys
    are "frame_inds", "clip_len", "frame_interval" and "num_clips".

    Args:
        clip_len (int): Frames of each sampled output clip.
        num_clips (int): Number of clips to be sampled. Default: 1.
        seed (int): The random seed used during test time. Default: 255.
    """

    def __init__(self,
                 clip_len,
                 num_clips=1,
                 p_interval=1,
                 seed=255,
                 **deprecated_kwargs):

        self.clip_len = clip_len
        self.num_clips = num_clips
        self.seed = seed
        self.p_interval = p_interval
        if not isinstance(p_interval, tuple):
            self.p_interval = (p_interval, p_interval)
        # if len(deprecated_kwargs):
            # warning_r0('[UniformSampleFrames] The following args has been deprecated: ')
            # for k, v in deprecated_kwargs.items():
            #     warning_r0(f'Arg name: {k}; Arg value: {v}')

    def _get_train_clips(self, num_frames, clip_len):
        """Uniformly sample indices for training clips.

        Args:
            num_frames (int): The number of frames.
            clip_len (int): The length of the clip.
        """
        allinds = []
        for clip_idx in range(self.num_clips):
            old_num_frames = num_frames
            pi = self.p_interval
            ratio = np.random.rand() * (pi[1] - pi[0]) + pi[0]
            num_frames = int(ratio * num_frames)
            off = np.random.randint(old_num_frames - num_frames + 1)

            if num_frames < clip_len:
                start = np.random.randint(0, num_frames)
                inds = np.arange(start, start + clip_len)
            elif clip_len <= num_frames < 2 * clip_len:
                basic = np.arange(clip_len)
                inds = np.random.choice(
                    clip_len + 1, num_frames - clip_len, replace=False)
                offset = np.zeros(clip_len + 1, dtype=np.int64)
                offset[inds] = 1
                offset = np.cumsum(offset)
                inds = basic + offset[:-1]
            else:
                bids = np.array(
                    [i * num_frames // clip_len for i in range(clip_len + 1)])
                bsize = np.diff(bids)
                bst = bids[:clip_len]
                offset = np.random.randint(bsize)
                inds = bst + offset

            inds = inds + off
            num_frames = old_num_frames

            allinds.append(inds)

        return np.concatenate(allinds)

    def _get_test_clips(self, num_frames, clip_len):
        """Uniformly sample indices for testing clips.

        Args:
            num_frames (int): The number of frames.
            clip_len (int): The length of the clip.
        """
        np.random.seed(self.seed)

        all_inds = []

        for i in range(self.num_clips):

            old_num_frames = num_frames
            pi = self.p_interval
            ratio = np.random.rand() * (pi[1] - pi[0]) + pi[0]
            num_frames = int(ratio * num_frames)
            off = np.random.randint(old_num_frames - num_frames + 1)

            if num_frames < clip_len:
                start_ind = i if num_frames < self.num_clips else i * num_frames // self.num_clips
                inds = np.arange(start_ind, start_ind + clip_len)
            elif clip_len <= num_frames < clip_len * 2:
                basic = np.arange(clip_len)
                inds = np.random.choice(clip_len + 1, num_frames - clip_len, replace=False)
                offset = np.zeros(clip_len + 1, dtype=np.int64)
                offset[inds] = 1
                offset = np.cumsum(offset)
                inds = basic + offset[:-1]
            else:
                bids = np.array([i * num_frames // clip_len for i in range(clip_len + 1)])
                bsize = np.diff(bids)
                bst = bids[:clip_len]
                offset = np.random.randint(bsize)
                inds = bst + offset

            all_inds.append(inds + off)
            num_frames = old_num_frames

        return np.concatenate(all_inds)

    def __call__(self, results):
        num_frames = results['total_frames']

        if results.get('test_mode', False):
            inds = self._get_test_clips(num_frames, self.clip_len)
        else:
            inds = self._get_train_clips(num_frames, self.clip_len)

        inds = np.mod(inds, num_frames)
        start_index = results['start_index']
        inds = inds + start_index

        if 'keypoint' in results:
            kp = results['keypoint']
            assert num_frames == kp.shape[1]
            num_person = kp.shape[0]
            num_persons = [num_person] * num_frames
            for i in range(num_frames):
                j = num_person - 1
                while j >= 0 and np.all(np.abs(kp[j, i]) < 1e-5):
                    j -= 1
                num_persons[i] = j + 1
            transitional = [False] * num_frames
            for i in range(1, num_frames - 1):
                if num_persons[i] != num_persons[i - 1]:
                    transitional[i] = transitional[i - 1] = True
                if num_persons[i] != num_persons[i + 1]:
                    transitional[i] = transitional[i + 1] = True
            inds_int = inds.astype(np.int)
            coeff = np.array([transitional[i] for i in inds_int])
            inds = (coeff * inds_int + (1 - coeff) * inds).astype(np.float32)

        results['frame_inds'] = inds.astype(np.int)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = None
        results['num_clips'] = self.num_clips
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'num_clips={self.num_clips}, '
                    f'seed={self.seed})')
        return repr_str


class PoseDecode:
    """Load and decode pose with given indices.

    Required keys are "keypoint", "frame_inds" (optional), "keypoint_score" (optional), added or modified keys are
    "keypoint", "keypoint_score" (if applicable).
    """

    @staticmethod
    def _load_kp(kp, frame_inds):
        return kp[:, frame_inds].astype(np.float32)

    @staticmethod
    def _load_kpscore(kpscore, frame_inds):
        return kpscore[:, frame_inds].astype(np.float32)

    def __call__(self, results):

        if 'frame_inds' not in results:
            results['frame_inds'] = np.arange(results['total_frames'])

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        offset = results.get('offset', 0)
        frame_inds = results['frame_inds'] + offset

        if 'keypoint_score' in results:
            results['keypoint_score'] = self._load_kpscore(results['keypoint_score'], frame_inds)

        if 'keypoint' in results:
            results['keypoint'] = self._load_kp(results['keypoint'], frame_inds)

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}()'
        return repr_str

class FormatGCNInput:
    """Format final skeleton shape to the given input_format. """

    def __init__(self, num_person=2, mode='zero'):
        self.num_person = num_person
        assert mode in ['zero', 'loop']
        self.mode = mode

    def __call__(self, results):
        """Performs the FormatShape formatting.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        keypoint = results['keypoint']
        if 'keypoint_score' in results:
            keypoint = np.concatenate((keypoint, results['keypoint_score'][..., None]), axis=-1)

        # M T V C
        if keypoint.shape[0] < self.num_person:
            pad_dim = self.num_person - keypoint.shape[0]
            pad = np.zeros((pad_dim, ) + keypoint.shape[1:], dtype=keypoint.dtype)
            keypoint = np.concatenate((keypoint, pad), axis=0)
            if self.mode == 'loop' and keypoint.shape[0] == 1:
                for i in range(1, self.num_person):
                    keypoint[i] = keypoint[0]

        elif keypoint.shape[0] > self.num_person:
            keypoint = keypoint[:self.num_person]

        M, T, V, C = keypoint.shape
        nc = results.get('num_clips', 1)
        assert T % nc == 0
        keypoint = keypoint.reshape((M, nc, T // nc, V, C)).transpose(1, 0, 2, 3, 4)
        results['keypoint'] = np.ascontiguousarray(keypoint)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(num_person={self.num_person}, mode={self.mode})'
        return repr_str

def assert_tensor_type(func: Callable) -> Callable:

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not isinstance(args[0].data, torch.Tensor):
            raise AttributeError(
                f'{args[0].__class__.__name__} has no attribute '
                f'{func.__name__} for type {args[0].datatype}')
        return func(*args, **kwargs)

    return wrapper


class DataContainer:
    """A container for any type of objects.
    Typically tensors will be stacked in the collate function and sliced along
    some dimension in the scatter function. This behavior has some limitations.
    1. All tensors have to be the same size.
    2. Types are limited (numpy array or Tensor).
    We design `DataContainer` and `MMDataParallel` to overcome these
    limitations. The behavior can be either of the following.
    - copy to GPU, pad all tensors to the same size and stack them
    - copy to GPU without stacking
    - leave the objects as is and pass it to the model
    - pad_dims specifies the number of last few dimensions to do padding
    """

    def __init__(self,
                 data: Union[torch.Tensor, np.ndarray],
                 stack: bool = False,
                 padding_value: int = 0,
                 cpu_only: bool = False,
                 pad_dims: int = 2):
        self._data = data
        self._cpu_only = cpu_only
        self._stack = stack
        self._padding_value = padding_value
        assert pad_dims in [None, 1, 2, 3]
        self._pad_dims = pad_dims

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({repr(self.data)})'

    def __len__(self) -> int:
        return len(self._data)

    @property
    def data(self) -> Union[torch.Tensor, np.ndarray]:
        return self._data

    @property
    def datatype(self) -> Union[Type, str]:
        if isinstance(self.data, torch.Tensor):
            return self.data.type()
        else:
            return type(self.data)

    @property
    def cpu_only(self) -> bool:
        return self._cpu_only

    @property
    def stack(self) -> bool:
        return self._stack

    @property
    def padding_value(self) -> int:
        return self._padding_value

    @property
    def pad_dims(self) -> int:
        return self._pad_dims

    @assert_tensor_type
    def size(self, *args, **kwargs) -> torch.Size:
        return self.data.size(*args, **kwargs)

    @assert_tensor_type
    def dim(self) -> int:
        return self.data.dim()


class Collect:
    """Collect data from the loader relevant to the specific task.

    This keeps the items in ``keys`` as it is, and collect items in
    ``meta_keys`` into a meta item called ``meta_name``.This is usually
    the last stage of the data loader pipeline.
    For example, when keys='imgs', meta_keys=('filename', 'label',
    'original_shape'), meta_name='img_metas', the results will be a dict with
    keys 'imgs' and 'img_metas', where 'img_metas' is a DataContainer of
    another dict with keys 'filename', 'label', 'original_shape'.

    Args:
        keys (Sequence[str]): Required keys to be collected.
        meta_name (str): The name of the key that contains meta information.
            This key is always populated. Default: "img_metas".
        meta_keys (Sequence[str]): Keys that are collected under meta_name.
            The contents of the ``meta_name`` dictionary depends on
            ``meta_keys``.
            By default this includes:

            - "filename": path to the image file
            - "label": label of the image file
            - "original_shape": original shape of the image as a tuple
                (h, w, c)
            - "img_shape": shape of the image input to the network as a tuple
                (h, w, c).  Note that images may be zero padded on the
                bottom/right, if the batch tensor is larger than this shape.
            - "pad_shape": image shape after padding
            - "flip_direction": a str in ("horiziontal", "vertival") to
                indicate if the image is fliped horizontally or vertically.
            - "img_norm_cfg": a dict of normalization information:
                - mean - per channel mean subtraction
                - std - per channel std divisor
                - to_rgb - bool indicating if bgr was converted to rgb
        nested (bool): If set as True, will apply data[x] = [data[x]] to all
            items in data. The arg is added for compatibility. Default: False.
    """

    def __init__(self,
                 keys,
                 meta_keys=('filename', 'label', 'original_shape', 'img_shape',
                            'pad_shape', 'flip_direction', 'img_norm_cfg'),
                 meta_name='img_metas',
                 nested=False):
        self.keys = keys
        self.meta_keys = meta_keys
        self.meta_name = meta_name
        self.nested = nested

    def __call__(self, results):
        """Performs the Collect formatting.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        data = {}
        for key in self.keys:
            data[key] = results[key]

        if len(self.meta_keys) != 0:
            meta = {}
            for key in self.meta_keys:
                meta[key] = results[key]
            data[self.meta_name] = DataContainer(meta, cpu_only=True)
        if self.nested:
            for k in data:
                data[k] = [data[k]]

        return data

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'keys={self.keys}, meta_keys={self.meta_keys}, '
                f'nested={self.nested})')


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    # if isinstance(data, Sequence) and not mmcv.is_str(data):
    #     return torch.tensor(data)
    if isinstance(data, int):
        return torch.LongTensor([data])
    if isinstance(data, float):
        return torch.FloatTensor([data])
    raise TypeError(f'type {type(data)} cannot be converted to tensor.')

class ToTensor:
    """Convert some values in results dict to `torch.Tensor` type in data
    loader pipeline.

    Args:
        keys (Sequence[str]): Required keys to be converted.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        """Performs the ToTensor formatting.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        for key in self.keys:
            results[key] = to_tensor(results[key])
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(keys={self.keys})'
    

class Preprocess_Module:

    def __init__(self, data_augmentation):
        self.transforms = []
        self.transforms.append(PreNormalize2D(img_shape=(320, 480), threshold=0.01,data_augmentation=data_augmentation))
        self.transforms.append(GenSkeFeat(dataset='coco', feats=['j']))
        # self.transforms.append(UniformSampleFrames(clip_len=200, num_clips=10))
        self.transforms.append(PoseDecode())
        self.transforms.append(FormatGCNInput(num_person=1))
        self.transforms.append(Collect(keys=['keypoint', 'label'], meta_keys=[]))
        self.transforms.append(ToTensor(keys=['keypoint']))


    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data
