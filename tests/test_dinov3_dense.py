import torch

from roadseg.train_dinov3 import DenseDINOv3, predict_logits


def test_dense_patch_geometry_and_frozen_encoder_gradient_boundary():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    model=DenseDINOv3(pretrained=False).train()
    assert not model.encoder.training
    assert model.encoder.num_prefix_tokens==5
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==7315
    image=torch.randn(1,3,64,64)
    patches=model.patch_features(image)
    assert patches.shape==(1,384,4,4)
    before=model.encoder.norm.weight.detach().clone()
    output=model(image)
    assert output.shape==(1,19,64,64)
    torch.nn.functional.cross_entropy(output,torch.zeros(1,64,64,dtype=torch.long)).backward()
    assert model.head.weight.grad is not None
    assert torch.isfinite(model.head.weight.grad).all()
    assert all(p.grad is None for p in model.encoder.parameters())
    torch.testing.assert_close(before,model.encoder.norm.weight,rtol=0,atol=0)


def test_tiled_inference_keeps_non_patch_divisible_image_size():
    torch.set_num_threads(4)
    model=DenseDINOv3(pretrained=False).eval()
    result=predict_logits(model,torch.randn(1,3,83,117),tile_size=64,stride=48,batch_size=2)
    assert result.shape==(1,19,83,117)
    assert torch.isfinite(result).all()
