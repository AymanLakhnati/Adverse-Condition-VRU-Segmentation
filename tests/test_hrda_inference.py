"""Check the two-level tiling port independently of dataset availability."""
import numpy as np
import torch

from src.hrda_inference import HRDA, crop_boxes, resize


def test_official_full_resolution_grid_covers_every_pixel():
    outer = list(crop_boxes(1080,1920,(1024,1024),(512,512)))
    assert len(outer) == 6
    coverage = np.zeros((1080,1920),dtype=np.uint8)
    for y1,y2,x1,x2 in outer:
        assert (y2-y1,x2-x1) == (1024,1024)
        assert all(v % 4 == 0 for v in (y1,y2,x1,x2))
        coverage[y1:y2,x1:x2] += 1
    assert np.all(coverage > 0)
    assert len(list(crop_boxes(1024,1024,(512,512),(256,256)))) == 9


def test_sequential_detail_logits_match_batched_reference():
    # Smaller grids exercise the same overlapping fusion without needing a GPU.
    torch.set_num_threads(4)
    torch.manual_seed(12)
    model = HRDA(outer_crop=128,detail_crop=64).eval()
    image = torch.randn(1,3,128,128)
    with torch.inference_mode():
        actual = model.encode_decode(image)
        low_features = model.backbone(resize(image,scale_factor=0.5))
        low = model.decode_head.head(low_features)
        attention = torch.sigmoid(model.decode_head.scale_attention(low_features))
        boxes = list(crop_boxes(128,128,(64,64),(32,32)))
        tiles = torch.cat([image[:,:,y1:y2,x1:x2] for y1,y2,x1,x2 in boxes])
        tile_logits = model.decode_head.head(model.backbone(tiles))
        high,counts = torch.zeros(1,19,32,32),torch.zeros(1,1,32,32)
        for i,(y1,y2,x1,x2) in enumerate(boxes):
            high[:,:,y1//4:y2//4,x1//4:x2//4] += tile_logits[i:i+1]
            counts[:,:,y1//4:y2//4,x1//4:x2//4] += 1
        expected = resize(resize(attention,scale_factor=2)*(high/counts) +
                          resize((1-attention)*low,scale_factor=2),size=(128,128))
    torch.testing.assert_close(actual,expected,rtol=3e-4,atol=3e-4)
