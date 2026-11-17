---
license: apache-2.0
language:
- en
pretty_name: Amazon ESCI products with multilingual-e5-small embeddings
task_categories:
- text-retrieval
- feature-extraction
size_categories:
- 100K<n<1M
tags:
- qdrant
- hybrid-search
- e-commerce
configs:
- config_name: products
  default: true
  data_files:
  - split: train
    path: data/*.parquet
- config_name: judgments
  data_files:
  - split: test
    path: judgments/*.parquet
---

# esci-products-multilingual-e5-small-embeddings

This dataset contains 164,900 product listings from the [Amazon Shopping Queries Dataset](https://github.com/amazon-science/esci-data)
(ESCI): every product judged for the US test queries of the dataset's small version. It also contains those
8,956 queries with their 181,701 relevance judgments, so you can measure how well a search setup ranks the products.

The `products` configuration holds one row per product:

| Column | Type | Description |
|---|---|---|
| `id` | string | Point ID in the Qdrant collection |
| `product_id` | string | Amazon product ID (ASIN) |
| `title` | string | Product title |
| `brand` | string | Brand, empty for about 5% of products |
| `color` | string | Color, empty for about 31% of products |
| `text` | string | The title, a `Brand:` line when the brand is known, and the bullet points |
| `vector` | list of float32 | multilingual-e5-small embedding, 384 dimensions, cosine |

The `judgments` configuration holds one row per query, with a list of its judged products and their labels:

```json
{"query_id": 2, "query": "!qscreen fence without holes", "judgments": [{"product_id": "B07DHX8YH2", "label": "E"}, {"product_id": "B07DS1YCRZ", "label": "S"}, {"product_id": "B07D7TBSGH", "label": "I"}]}
```

The same judgments are also in `queries.jsonl` at the root of the repository, with each query's judgments as a
mapping from product ID to label.

The labels are `E` for exact, `S` for substitute, `C` for complement, and `I` for irrelevant. For nDCG, the ESCI
paper uses gains of 1.0, 0.1, 0.01, and 0.0. Products a search returns without a judgment count as irrelevant.

## Generation process

The embeddings were generated with the [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small)
model from the `text` field with the `passage: ` prefix. Embed your search queries with the `query: ` prefix instead.
The model reads up to 512 tokens, so for about 3% of products the embedding covers only the start of the text.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/multilingual-e5-small")
text = "Nature Valley Sweet and Salty Nut Granola Bars, Cashew, 12 ct, 14.8 oz\nBrand: Nature Valley"
embeddings = model.encode([f"passage: {text}"], normalize_embeddings=True)
query_embedding = model.encode(
    ["cashew granola bars"], prompt="query: ", normalize_embeddings=True
)
```

## Using the dataset with Qdrant

The quickest way is to restore the ready-made [snapshot](https://snapshots.qdrant.io/esci-products.snapshot),
which includes the payload indexes. It requires Qdrant 1.20 or later:

```http
PUT /collections/esci-products/snapshots/recover
{
  "location": "https://snapshots.qdrant.io/esci-products.snapshot"
}
```

To load the rows yourself, create the collection with a vector named `dense`, upload the rows, then create the
payload indexes. The full-text index on `text` uses BM25 scoring, which also requires Qdrant 1.20 or later:

```python
from datasets import load_dataset
from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")
client.create_collection(
    "esci-products",
    vectors_config={
        "dense": models.VectorParams(size=384, distance=models.Distance.COSINE)
    },
)

products = load_dataset(
    "Qdrant/esci-products-multilingual-e5-small-embeddings", "products", split="train"
)
for batch in products.iter(batch_size=256):
    payload_fields = [name for name in batch if name not in ("id", "vector")]
    client.upsert(
        "esci-products",
        points=[
            models.PointStruct(
                id=batch["id"][i],
                vector={"dense": batch["vector"][i]},
                payload={
                    name: batch[name][i]
                    for name in payload_fields
                    if batch[name][i] is not None
                },
            )
            for i in range(len(batch["id"]))
        ],
    )

for field in ("product_id", "brand", "color"):
    client.create_payload_index(
        "esci-products", field, models.PayloadSchemaType.KEYWORD
    )
client.create_payload_index(
    "esci-products",
    "text",
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

judgments = load_dataset(
    "Qdrant/esci-products-multilingual-e5-small-embeddings", "judgments", split="test"
)
```

Pass `using="dense"` when you query the vector.

## License and attribution

The Shopping Queries Dataset is released by Amazon Science under the [Apache 2.0 license](https://github.com/amazon-science/esci-data/blob/main/LICENSE).
This dataset is a subset of it with embeddings added. If you use it in research, cite the paper:

> Chandan K. Reddy, Lluís Màrquez, Fran Valero, Nikhil Rao, Hugo Zaragoza, Sambaran Bandyopadhyay, Arnab Biswas,
> Anlu Xing, and Karthik Subbian. 2022. [Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving
> Product Search](https://arxiv.org/abs/2206.06588). arXiv:2206.06588.
