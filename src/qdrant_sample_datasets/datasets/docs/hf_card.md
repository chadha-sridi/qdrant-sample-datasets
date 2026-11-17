---
license: cc-by-4.0
language:
- en
pretty_name: Qdrant documentation with multilingual-e5-small embeddings
task_categories:
- text-retrieval
- feature-extraction
size_categories:
- 1K<n<10K
tags:
- qdrant
- hybrid-search
---

# qdrant-docs-multilingual-e5-small-embeddings

This dataset contains the Qdrant documentation, courses, and articles from [qdrant.tech](https://qdrant.tech),
as of October 25, 2026, split into 3,593 chunks by page section. A section that fits in 480 tokens becomes one
chunk, and larger sections are split further. Code examples are kept in Python only.

Each row holds a chunk's ID, its payload, and a 384-dimensional embedding:

| Column | Type | Description |
|---|---|---|
| `id` | string | Point ID in the Qdrant collection |
| `url` | string | Page URL, relative to qdrant.tech |
| `title` | string | Page title |
| `section` | string | Headings above the chunk, joined with " > ". Empty for page intros |
| `content_type` | string | `documentation`, `course`, or `article` |
| `has_code` | bool | Whether the chunk contains a code block |
| `text` | string | A heading line, then the chunk in Markdown |
| `vector` | list of float32 | multilingual-e5-small embedding, 384 dimensions, cosine |

## Generation process

The embeddings were generated with the [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small)
model from the `text` field with fenced code blocks, Markdown images, and link URLs removed, using the `passage: `
prefix. Embed your search queries with the `query: ` prefix instead:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/multilingual-e5-small")
text = "Final Project: Production-Ready Documentation Search Engine > Build Steps > Step 3: Parse and Chunk Documents"
embeddings = model.encode([f"passage: {text}"], normalize_embeddings=True)
query_embedding = model.encode(
    ["how to chunk documentation"], prompt="query: ", normalize_embeddings=True
)
```

## Using the dataset with Qdrant

The quickest way is to restore the ready-made [snapshot](https://snapshots.qdrant.io/qdrant-docs-2026-10-25.snapshot),
which includes the payload indexes. It requires Qdrant 1.20 or later:

```http
PUT /collections/qdrant-docs/snapshots/recover
{
  "location": "https://snapshots.qdrant.io/qdrant-docs-2026-10-25.snapshot"
}
```

To load the rows yourself, create the collection with a vector named `dense`, upload the rows, then create the
payload indexes. The full-text index on `text` uses BM25 scoring, which also requires Qdrant 1.20 or later:

```python
from datasets import load_dataset
from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")
client.create_collection(
    "qdrant-docs",
    vectors_config={
        "dense": models.VectorParams(size=384, distance=models.Distance.COSINE)
    },
)

dataset = load_dataset(
    "Qdrant/qdrant-docs-multilingual-e5-small-embeddings", split="train"
)
for batch in dataset.iter(batch_size=256):
    payload_fields = [name for name in batch if name not in ("id", "vector")]
    client.upsert(
        "qdrant-docs",
        points=[
            models.PointStruct(
                id=batch["id"][i],
                vector={"dense": batch["vector"][i]},
                payload={name: batch[name][i] for name in payload_fields},
            )
            for i in range(len(batch["id"]))
        ],
    )

for field in ("url", "content_type"):
    client.create_payload_index("qdrant-docs", field, models.PayloadSchemaType.KEYWORD)
client.create_payload_index("qdrant-docs", "has_code", models.PayloadSchemaType.BOOL)
client.create_payload_index(
    "qdrant-docs",
    "text",
    models.TextIndexParams(
        type=models.TextIndexType.TEXT,
        tokenizer=models.TokenizerType.CODE,
        lowercase=True,
        stemmer=models.SnowballParams(
            type=models.Snowball.SNOWBALL, language=models.SnowballLanguage.ENGLISH
        ),
        stopwords=models.Language.ENGLISH,
        scoring=models.TextScoringParams(type=models.TextScoringType.BM25),
    ),
)
```

Pass `using="dense"` when you query the vector.

## License

The content belongs to Qdrant and is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
