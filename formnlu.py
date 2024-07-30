# coding=utf-8
'''
Reference: https://huggingface.co/datasets/nielsr/funsd/blob/main/funsd.py
'''
import json
import os

import datasets

from image_utils import load_image, normalize_bbox


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@article{Jaume2019FUNSDAD,
  title={FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents},
  author={Guillaume Jaume and H. K. Ekenel and J. Thiran},
  journal={2019 International Conference on Document Analysis and Recognition Workshops (ICDARW)},
  year={2019},
  volume={2},
  pages={1-6}
}
"""

_DESCRIPTION = """\
https://guillaumejaume.github.io/FUNSD/
"""


class Formnluconfig(datasets.BuilderConfig):
    """BuilderConfig for FUNSD"""

    def __init__(self, **kwargs):
        """BuilderConfig for FUNSD.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(Formnluconfig, self).__init__(**kwargs)


class Formnlu(datasets.GeneratorBasedBuilder):
    """Conll2003 dataset."""

    BUILDER_CONFIGS = [
        Formnluconfig(name="formnlu", version=datasets.Version("1.0.0"), description="FORMNLU dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "global_id": datasets.Sequence(datasets.Value("string")),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "bbox": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "labels": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=["company name", \
                                    "substantial holder name", \
                                    "holder ACN/ARSN",\
                                    "There was a change in the interests of the substantial holder on",\
                                    "The previous notice was dated",\
                                    "The previous notice was given to the company on",\
                                    "class of securities",\
                                    "Previous notice Person's notes",\
                                    "Previous notice Voting power",\
                                    "Present notice Person's votes",\
                                    "Present notice Voting power",\
                                    "company ACN/ARSN",\
                                    "NULL"]
                        )
                    ),
                    "image": datasets.Array3D(shape=(3, 224, 224), dtype="uint8"),
                    "image_path": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": f"/content/data/train_data"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={"filepath": f"/content/data/val_data"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": f"/content/data/test_data"}
            ),
        ]

    def get_line_bbox(self, bboxs):
        x = [bboxs[i][j] for i in range(len(bboxs)) for j in range(0, len(bboxs[i]), 2)]
        y = [bboxs[i][j] for i in range(len(bboxs)) for j in range(1, len(bboxs[i]), 2)]

        x0, y0, x1, y1 = min(x), min(y), max(x), max(y)

        assert x1 >= x0 and y1 >= y0
        bbox = [[x0, y0, x1, y1] for _ in range(len(bboxs))]
        return bbox

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        ann_dir = os.path.join(filepath, "annotations")
        img_dir = os.path.join(filepath, "images")
        for guid, file in enumerate(sorted(os.listdir(ann_dir))):
            tokens = []
            bboxes = []
            labels = []
            global_ids = []
            file_name = file.split('.')[0]
            file_path = os.path.join(ann_dir, file)
            with open(file_path, "r", encoding="utf8") as f:
                data = json.load(f)
            image_path = os.path.join(img_dir, file)
            image_path = image_path.replace("json", "png")
            image, size = load_image(image_path)
            for item in data["form"]:
                cur_line_bboxes = []
                words, label = item["words"], item["label"]
                words = [w for w in words if w["text"].strip() != ""]
                if len(words) == 0:
                    continue
                # if label == "NULL":
                    # for w in words:
                        # tokens.append(w["text"])
                        # labels.append("O")
                        # cur_line_bboxes.append(normalize_bbox(w["box"], size))
                # else:
                tokens.append(words[0]["text"])
                labels.append(label)
                cur_line_bboxes.append(normalize_bbox(words[0]["box"], size))
                # for w in words[1:]:
                    # tokens.append(w["text"])
                    # labels.append("I-" + label.upper())
                    # cur_line_bboxes.append(normalize_bbox(w["box"], size))
                # by default: --segment_level_layout 1
                # if do not want to use segment_level_layout, comment the following line
                cur_line_bboxes = self.get_line_bbox(cur_line_bboxes)
                # box = normalize_bbox(item["box"], size)
                # cur_line_bboxes = [box for _ in range(len(words))]
                bboxes.extend(cur_line_bboxes)
                global_ids.append(item['global_id'])
            yield guid, {"id": str(guid), "tokens": tokens, "bbox": bboxes, "labels": labels,
                         "image": image, "image_path": image_path, 'global_id':global_ids}
