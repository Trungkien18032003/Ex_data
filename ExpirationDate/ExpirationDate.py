import json
import os
import datasets

# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@article{seker2022generalized, title={A generalized framework for recognition of expiration dates on product packages using fully convolutional networks}, author={Seker, Ahmet Cagatay and Ahn, Sang Chul}, journal={Expert Systems with Applications}, pages={117310}, year={2022}, publisher={Elsevier} }
"""

_DESCRIPTION = """\
The dataset for Date detection in the proposed framework aims to provide annotated images that are relevant for training and evaluating models tasked with detecting dates within product labels or similar contexts.
"""

_HOMEPAGE = "https://acseker.github.io/ExpDateWebsite/"

_LICENSE = "https://licenses.nuget.org/AFL-3.0"

_URLs = {
    "products_synth": "https://huggingface.co/datasets/dimun/ExpirationDate/resolve/main/Products-Synth.zip",
    "products_real": "https://huggingface.co/datasets/dimun/ExpirationDate/resolve/main/Products-Real.zip",
}


def has_extension(file_path, extensions):
    _, file_extension = os.path.splitext(file_path)
    return file_extension.lower() in extensions


logger = datasets.logging.get_logger(__name__)


class ExpirationDate(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.1")
    CATEGORIES = ["prod", "date", "due", "code"]

    def _info(self):
        features = datasets.Features(
            {
                "id": datasets.Value("string"),
                "transcriptions": datasets.Sequence(datasets.Value("string")),
                "bboxes_block": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                "categories": datasets.Sequence(datasets.features.ClassLabel(names=self.CATEGORIES)),
                "image_path": datasets.Value("string"),
                "width": datasets.Value("int32"),
                "height": datasets.Value("int32")
            }
        )

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # Features/targets of the dataset
            features=features,
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract files
        # based on the provided URLs

        archive_path = dl_manager.download_and_extract(_URLs)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(archive_path["products_synth"], "Products-Synth/"),
                    "split": "",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(archive_path["products_real"], "Products-Real/"),
                    "split": "evaluation",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(archive_path["products_real"], "Products-Real/"),
                    # Using train of products real as test
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        logger.info(
            f"⏳ Generating examples from = {filepath} to the split {split}")
        ann_file = os.path.join(filepath, split, "annotations.json")

        # get json
        with open(ann_file, "r", encoding="utf8") as f:
            features_map = json.load(f)

        img_dir = os.path.join(filepath, split, "images")
        img_listdir = os.listdir(img_dir)

        for guid, filename in enumerate(img_listdir):
            if filename.endswith(".jpg"):
                image_features = features_map[filename]
                image_ann = image_features.get("ann")

                transcriptions = [box.get("transcription", "")
                                  for box in image_ann]
                bboxes_block = [box.get("bbox") for box in image_ann]
                categories = [box.get("cls") if box.get(
                    "cls") in self.CATEGORIES else "date" for box in image_ann]

                # get image
                image_path = os.path.join(img_dir, filename)

                yield guid, {
                    "id": filename,
                    "transcriptions": transcriptions,
                    "bboxes_block": bboxes_block,
                    "categories": categories,
                    "image_path": image_path,
                    "width": image_features.get("width"),
                    "height": image_features.get("height"),
                }
