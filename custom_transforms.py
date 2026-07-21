import mmcv
import mmengine.fileio as fileio
import numpy as np
from mmcv.transforms import LoadImageFromFile
from mmengine.registry import TRANSFORMS
from typing import Optional


@TRANSFORMS.register_module()
class LoadCDImagesFromFile(LoadImageFromFile):
    """Load both images from file for change detection datasets.
    
    This transform loads both img1 and img2 from img1_path and img2_path
    used in change detection datasets.
    
    Note: We inherit from mmcv's LoadImageFromFile to reuse its initialization
    parameters (color_type, channel_order, etc.), but we override transform()
    to load both images simultaneously.
    """
    
    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load both images.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded images and meta information.
        """
        
        def _load_image(filename: str) -> np.ndarray:
            """Helper function to load a single image."""
            try:
                if self.file_client_args is not None:
                    file_client = fileio.FileClient.infer_client(
                        self.file_client_args, filename)
                    img_bytes = file_client.get(filename)
                else:
                    img_bytes = fileio.get(
                        filename, backend_args=self.backend_args)
                img = mmcv.imfrombytes(
                    img_bytes, flag=self.color_type, backend=self.imdecode_backend)
            except Exception as e:
                if self.ignore_empty:
                    return None
                else:
                    raise e
            # in some cases, images are not read successfully, the img would be
            # `None`, refer to https://github.com/open-mmlab/mmpretrain/issues/1427
            assert img is not None, f'failed to load image: {filename}'
            if self.to_float32:
                img = img.astype(np.float32)
            return img

        # Load img1
        img1 = _load_image(results['img1_path'])
        if img1 is None:
            return None
        
        results['img'] = img1
        results['img_shape'] = img1.shape[:2]
        results['ori_shape'] = img1.shape[:2]
        
        # Load img2
        img2 = _load_image(results['img2_path'])
        if img2 is None:
            return None
        
        results['img2'] = img2
        results['img2_shape'] = img2.shape[:2]
        results['img2_ori_shape'] = img2.shape[:2]
        
        return results


@TRANSFORMS.register_module()
class RandomDiscreteRotate90:
    """Uniformly rotate image and segmentation maps by 0/90/180/270 degrees."""

    def __init__(self, choices=(0, 1, 2, 3)):
        self.choices = tuple(int(value) for value in choices)
        if not self.choices or any(value not in (0, 1, 2, 3) for value in self.choices):
            raise ValueError("choices must be a non-empty subset of (0, 1, 2, 3)")

    def transform(self, results: dict) -> dict:
        k = int(np.random.choice(self.choices))
        if k:
            results['img'] = np.rot90(results['img'], k=k, axes=(0, 1)).copy()
            for key in results.get('seg_fields', []):
                results[key] = np.rot90(results[key], k=k, axes=(0, 1)).copy()
        results['discrete_rotation_k'] = k
        results['img_shape'] = results['img'].shape[:2]
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(choices={self.choices})'


@TRANSFORMS.register_module()
class MildBrightnessContrastSaturation:
    """Image-only mild color jitter without hue changes.

    Brightness, contrast, and saturation factors are sampled independently
    from the configured ranges.  The three operations are applied in a random
    order when the transform is activated.  Segmentation annotations are never
    touched.
    """

    def __init__(
            self,
            prob=0.8,
            brightness=(0.8, 1.2),
            contrast=(0.8, 1.2),
            saturation=(0.8, 1.2)):
        self.prob = float(prob)
        if not 0.0 <= self.prob <= 1.0:
            raise ValueError('prob must be in [0, 1]')
        self.brightness = self._validate_range(brightness, 'brightness')
        self.contrast = self._validate_range(contrast, 'contrast')
        self.saturation = self._validate_range(saturation, 'saturation')

    @staticmethod
    def _validate_range(value, name):
        if len(value) != 2 or float(value[0]) <= 0 or float(value[1]) < float(value[0]):
            raise ValueError(f'{name} must be a positive [min, max] range')
        return float(value[0]), float(value[1])

    @staticmethod
    def _brightness(image, factor):
        return image * factor

    @staticmethod
    def _contrast(image, factor):
        mean = image.mean(axis=(0, 1), keepdims=True)
        return (image - mean) * factor + mean

    @staticmethod
    def _saturation(image, factor):
        gray = image.mean(axis=2, keepdims=True)
        return (image - gray) * factor + gray

    def transform(self, results: dict) -> dict:
        if np.random.random() >= self.prob:
            results['mild_color_jitter'] = None
            return results
        factors = dict(
            brightness=float(np.random.uniform(*self.brightness)),
            contrast=float(np.random.uniform(*self.contrast)),
            saturation=float(np.random.uniform(*self.saturation)),
        )
        operations = list(factors)
        np.random.shuffle(operations)
        original_dtype = results['img'].dtype
        image = results['img'].astype(np.float32)
        for operation in operations:
            image = getattr(self, f'_{operation}')(image, factors[operation])
        image = np.clip(image, 0.0, 255.0)
        if np.issubdtype(original_dtype, np.integer):
            image = np.rint(image)
        results['img'] = image.astype(original_dtype)
        results['mild_color_jitter'] = dict(
            factors=factors,
            order=operations,
        )
        return results

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(prob={self.prob}, '
            f'brightness={self.brightness}, contrast={self.contrast}, '
            f'saturation={self.saturation})')
