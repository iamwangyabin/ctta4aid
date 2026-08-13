# Vendored from liyih/IAPL commit a173e7783bbafaa00d60e6e31774a0bc14411a23.
# Upstream did not publish a software license at the time of import.
from .clip_models import CLIPModel

VALID_NAMES = [
    'CLIP:ViT-B/32',
    'CLIP:ViT-B/16',
    'CLIP:ViT-L/14',
]

def build_model(args):
    if args.backbone.startswith("CLIP:"):
        assert args.backbone in VALID_NAMES
        return CLIPModel(args)
        # return SeArModel(args)
