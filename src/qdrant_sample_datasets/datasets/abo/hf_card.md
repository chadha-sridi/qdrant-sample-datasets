---
license: cc-by-4.0
language:
- en
pretty_name: Amazon Berkeley Objects products with clip-ViT-B-32 image embeddings
task_categories:
- image-feature-extraction
- feature-extraction
size_categories:
- 10K<n<100K
tags:
- qdrant
- multimodal
- e-commerce
---

# amazon-berkeley-objects-clip-ViT-B-32-embeddings

This dataset contains 44,932 products from the [Amazon Berkeley Objects](https://amazon-berkeley-objects.s3.amazonaws.com/index.html)
(ABO) dataset, each with a CLIP embedding of its main image. Because CLIP maps text and images into the same space,
you can find products by describing them in words or by showing a similar image.

The collection keeps ABO products that have an English name and a photo as their main image. Products that share a
main image, usually size or color variants, are kept once, under the shortest of their names. Phone cases make up
half of the source catalog, so 1,000 of them are kept, chosen at random.

| Column | Type | Description |
|---|---|---|
| `id` | string | Point ID in the Qdrant collection |
| `item_id` | string | ABO item ID |
| `name` | string | Product name in English |
| `brand` | string | Brand, empty for about 7% of products |
| `color` | string | Color, empty for about 43% of products |
| `product_type` | string | ABO product type, such as `CHAIR` |
| `country` | string | Marketplace of the listing, such as `US` |
| `image_url` | string | The main image, at most 256 pixels on the longest side |
| `vector` | list of float32 | clip-ViT-B-32 image embedding, 512 dimensions, cosine |

## Generation process

The embeddings were generated with the [clip-ViT-B-32](https://huggingface.co/sentence-transformers/clip-ViT-B-32)
model from the image at `image_url`. Encode text queries with the same model to search the images:

```python
from PIL import Image
from sentence_transformers import SentenceTransformer

image_path = "f3de0cfd.jpg"

model = SentenceTransformer("clip-ViT-B-32")
embedding = model.encode(Image.open(image_path))
query_embedding = model.encode("a red travel case")
```

On Qdrant Cloud, the same models are available through Cloud Inference as `qdrant/clip-vit-b-32-vision` and
`qdrant/clip-vit-b-32-text`.

## Using the dataset with Qdrant

The quickest way is to restore the ready-made [snapshot](https://snapshots.qdrant.io/amazon-berkeley-objects.snapshot),
which includes the payload indexes. It requires Qdrant 1.20 or later:

```http
PUT /collections/amazon-berkeley-objects/snapshots/recover
{
  "location": "https://snapshots.qdrant.io/amazon-berkeley-objects.snapshot"
}
```

To load the rows yourself, create the collection with a single unnamed vector, upload the rows, then create the
payload indexes. The full-text index on `name` uses BM25 scoring, which also requires Qdrant 1.20 or later:

```python
from datasets import load_dataset
from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")
client.create_collection(
    "amazon-berkeley-objects",
    vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE),
)

dataset = load_dataset(
    "Qdrant/amazon-berkeley-objects-clip-ViT-B-32-embeddings", split="train"
)
for batch in dataset.iter(batch_size=256):
    payload_fields = [name for name in batch if name not in ("id", "vector")]
    client.upsert(
        "amazon-berkeley-objects",
        points=[
            models.PointStruct(
                id=batch["id"][i],
                vector=batch["vector"][i],
                payload={
                    name: batch[name][i]
                    for name in payload_fields
                    if batch[name][i] is not None
                },
            )
            for i in range(len(batch["id"]))
        ],
    )

for field in ("item_id", "brand", "color", "product_type", "country"):
    client.create_payload_index(
        "amazon-berkeley-objects", field, models.PayloadSchemaType.KEYWORD
    )
client.create_payload_index(
    "amazon-berkeley-objects",
    "name",
    models.TextIndexParams(
        type=models.TextIndexType.TEXT,
        tokenizer=models.TokenizerType.WORD,
        lowercase=True,
        stemmer=models.SnowballParams(
            type=models.Snowball.SNOWBALL, language=models.SnowballLanguage.ENGLISH
        ),
        stopwords=models.Language.ENGLISH,
        scoring=models.TextScoringParams(type=models.TextScoringType.BM25),
    ),
)
```

## License and attribution

ABO is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Credit for the data, including the
images, goes to Amazon.com. Credit for building the dataset goes to its Amazon and UC Berkeley authors. This dataset
is a modified subset of the ABO listings, as described above, with a CLIP embedding of each main image added.

> Jasmine Collins, Shubham Goel, Kenan Deng, Achleshwar Luthra, Leon Xu, Erhan Gundogdu, Xi Zhang, Tomas F. Yago
> Vicente, Thomas Dideriksen, Himanshu Arora, Matthieu Guillaumin, and Jitendra Malik. 2022. [ABO: Dataset and
> Benchmarks for Real-World 3D Object Understanding](https://arxiv.org/abs/2110.06199). In Proceedings of the IEEE/CVF
> Conference on Computer Vision and Pattern Recognition (CVPR).
