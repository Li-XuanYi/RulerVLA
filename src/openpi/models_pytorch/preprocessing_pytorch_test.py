from types import SimpleNamespace

import pytest
import torch

from openpi.models_pytorch import preprocessing_pytorch


@pytest.mark.parametrize("input_layout", ["channels_last", "channels_first"])
def test_preprocess_observation_outputs_channels_first(input_layout: str):
    batch_size = 2
    image = torch.arange(batch_size * 224 * 224 * 3, dtype=torch.float32).reshape(batch_size, 224, 224, 3)
    input_image = image.permute(0, 3, 1, 2) if input_layout == "channels_first" else image
    images = {key: input_image.clone() for key in preprocessing_pytorch.IMAGE_KEYS}
    image_masks = {key: torch.ones(batch_size, dtype=torch.bool) for key in preprocessing_pytorch.IMAGE_KEYS}
    observation = SimpleNamespace(
        images=images,
        image_masks=image_masks,
        state=torch.zeros(batch_size, 32),
        tokenized_prompt=torch.zeros(batch_size, 200, dtype=torch.int64),
        tokenized_prompt_mask=torch.ones(batch_size, 200, dtype=torch.bool),
        token_ar_mask=None,
        token_loss_mask=None,
    )

    processed = preprocessing_pytorch.preprocess_observation_pytorch(observation, train=False)

    for output_image in processed.images.values():
        assert output_image.shape == (batch_size, 3, 224, 224)
        torch.testing.assert_close(output_image, image.permute(0, 3, 1, 2))
