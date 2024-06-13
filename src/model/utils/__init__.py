from .confidence import cal_confidence_batch
from .stop import DepictQAStop

VISION_TAGS = {
    "pos": {"img": "<image>"},
    "sov": {
        "img": "<Img>",
        "ref": "<Img-Reference>",
        "A": "<Img-A>",
        "B": "<Img-B>",
    },
    "eov": {
        "img": "</Img>",
        "ref": "</Img-Reference>",
        "A": "</Img-A>",
        "B": "</Img-B>",
    },
}
