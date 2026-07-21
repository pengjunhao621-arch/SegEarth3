import numpy as np

from mmengine.registry import TRANSFORMS

import custom_transforms  # noqa: F401


def test_custom_prompt_transforms_build_as_callables():
    rotate = TRANSFORMS.build(
        dict(type="RandomDiscreteRotate90", choices=(0, 1, 2, 3))
    )
    color = TRANSFORMS.build(
        dict(
            type="MildBrightnessContrastSaturation",
            prob=0.8,
            brightness=(0.8, 1.2),
            contrast=(0.8, 1.2),
            saturation=(0.8, 1.2),
        )
    )

    assert callable(rotate)
    assert callable(color)


def test_discrete_rotation_keeps_image_and_segmentation_aligned():
    image = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    segmentation = np.arange(2 * 3, dtype=np.uint8).reshape(2, 3)
    transform = TRANSFORMS.build(
        dict(type="RandomDiscreteRotate90", choices=(1,))
    )

    output = transform(
        dict(
            img=image.copy(),
            gt_seg_map=segmentation.copy(),
            seg_fields=["gt_seg_map"],
        )
    )

    np.testing.assert_array_equal(output["img"], np.rot90(image, 1, (0, 1)))
    np.testing.assert_array_equal(
        output["gt_seg_map"], np.rot90(segmentation, 1, (0, 1))
    )
    assert output["img_shape"] == (3, 2)
    assert output["discrete_rotation_k"] == 1


def test_identity_color_jitter_never_changes_segmentation():
    image = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3)
    segmentation = np.arange(3 * 4, dtype=np.uint8).reshape(3, 4)
    transform = TRANSFORMS.build(
        dict(
            type="MildBrightnessContrastSaturation",
            prob=1.0,
            brightness=(1.0, 1.0),
            contrast=(1.0, 1.0),
            saturation=(1.0, 1.0),
        )
    )

    output = transform(
        dict(
            img=image.copy(),
            gt_seg_map=segmentation.copy(),
            seg_fields=["gt_seg_map"],
        )
    )

    np.testing.assert_array_equal(output["img"], image)
    np.testing.assert_array_equal(output["gt_seg_map"], segmentation)
    assert sorted(output["mild_color_jitter"]["order"]) == [
        "brightness",
        "contrast",
        "saturation",
    ]
