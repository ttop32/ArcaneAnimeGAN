# based on https://github.com/bryandlee/animegan2-pytorch/blob/25d7b017267208dfaf34026aa3425e518372aa2f/hubconf.py

import torch


def generator(pretrained=True, device="cpu", progress=True, check_hash=True):
    from model import Generator

    device = torch.device(device)
    model = Generator().to(device)
    ckpt_url = "https://github.com/ttop32/ArcaneAnimeGAN/raw/main/weights/arcaneanimegan_0.1.pt"

    if pretrained:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(
                ckpt_url, map_location=device, progress=progress, check_hash=check_hash,
            )
        )
    return model


def face2paint(device="cpu", size=256):
    from PIL import Image
    from torchvision.transforms.functional import to_pil_image, to_tensor

    def face2paint(
        model: torch.nn.Module,
        img: Image.Image,
        size: int = size,
        device: str = device,
    ) -> Image.Image:
        w, h = img.size
        s = min(w, h)
        img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        img = img.resize((size, size), Image.LANCZOS)

        with torch.no_grad():
            input = to_tensor(img).unsqueeze(0) * 2.0 - 1
            output = model(input.to(device)).cpu()[0]
            output = to_pil_image((output * 0.5 + 0.5).clip(0, 1))
        return output

    return face2paint