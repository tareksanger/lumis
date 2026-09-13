"""FAISS behavior tested on tiny local vectors with no embedding service."""

from unittest.mock import Mock

from lumis.core.document import Chunk
from lumis.embedding.base_embedding import BaseEmbeddingModel
from lumis.storage import BaseVectorDB, FaissVectorDB

import numpy as np
import pytest


def vec(*values):
    return np.array(values, dtype=np.float32)


@pytest.fixture
def embedding():
    model = Mock(spec=BaseEmbeddingModel)
    model.dimension = 2
    vectors = {"east": vec(1, 0), "north": vec(0, 1), "west": vec(-1, 0)}
    model.embed.side_effect = lambda text: vectors[text] if isinstance(text, str) else np.vstack([vectors[t] for t in text])
    return model


@pytest.fixture
def db(embedding):
    return FaissVectorDB(embedding)


def chunk(name, *vector):
    return Chunk(doc_id=name, content=name, embedding=vec(*vector) if vector else None)


def test_base_requires_implementation():
    with pytest.raises(TypeError, match="abstract"):
        BaseVectorDB()


@pytest.mark.parametrize(
    "method,args",
    [
        ("aadd_chunks", ([],)),
        ("aadd_chunk", (chunk("east"),)),
        ("aadd_texts", ([],)),
        ("aadd_text", ("east",)),
        ("asearch", ("east",)),
    ],
)
async def test_base_async_methods_fail_explicitly(db, method, args):
    with pytest.raises(NotImplementedError):
        await getattr(BaseVectorDB, method)(db, *args)


def test_base_clear_fails_explicitly(db):
    with pytest.raises(NotImplementedError, match="clear"):
        db.clear()


def test_rejects_unknown_index(embedding):
    with pytest.raises(ValueError, match="Unsupported index type"):
        FaissVectorDB(embedding, index_type="unknown")


def test_empty_add_is_noop(db, embedding):
    db.add_chunks([])
    embedding.embed.assert_not_called()
    assert db.index.ntotal == 0
    assert db.next_id == 0


def test_precomputed_vectors_bypass_embedding_and_l2_search_orders_neighbors(db, embedding):
    chunks = [chunk("west", -1, 0), chunk("east", 1, 0), chunk("north", 0, 1)]
    db.add_chunks(chunks)
    embedding.embed.assert_not_called()
    assert db.search("east", k=10) == [chunks[1], chunks[2], chunks[0]]
    assert db.index.ntotal == 3
    assert db.next_id == 3


def test_mixed_batch_embeds_only_missing_vectors_in_order(db, embedding):
    chunks = [chunk("east", 1, 0), chunk("north"), chunk("west")]
    db.add_chunks(chunks)
    embedding.embed.assert_called_once_with(["north", "west"])
    assert db.search("north", k=1) == [chunks[1]]


def test_single_row_precomputed_vector_is_supported(db):
    item = Chunk(content="east", embedding=np.array([[1, 0]], dtype=np.float32))
    db.add_chunk(item)
    assert db.search("east", k=1) == [item]


@pytest.mark.parametrize("value", [vec(1, 2, 3), np.ones((2, 2), dtype=np.float32)])
def test_invalid_precomputed_vectors_leave_database_empty(db, value):
    with pytest.raises(ValueError, match="shape|dimension"):
        db.add_chunk(Chunk(content="invalid", embedding=value))
    assert db.index.ntotal == 0
    assert db.id_map == {}
    assert db.next_id == 0


def test_generated_dimension_mismatch_is_rejected(db, embedding):
    embedding.embed.side_effect = None
    embedding.embed.return_value = np.ones((1, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="dimension mismatch"):
        db.add_chunk(chunk("east"))
    assert db.index.ntotal == 0


def test_query_dimension_mismatch_is_rejected(db, embedding):
    embedding.embed.side_effect = None
    embedding.embed.return_value = vec(1, 2, 3)
    with pytest.raises(ValueError, match="dimension mismatch"):
        db.search("anything")


def test_search_empty_database(db):
    assert db.search("east") == []


def test_search_respects_fetch_and_result_limits(db):
    db.add_chunks([chunk("east", 1, 0), chunk("north", 0, 1), chunk("west", -1, 0)])
    assert len(db.search("east", k=1)) == 1
    assert len(db.search("east", k=3, fetch_k=2)) == 2


def test_text_helpers_preserve_metadata_and_allow_short_metadata_list(db):
    db.add_texts(["east", "north"], [{"source": "first"}])
    db.add_text("west", {"source": "third"})
    chunks = list(db.id_map.values())
    assert [c.content for c in chunks] == ["east", "north", "west"]
    assert [c.metadata for c in chunks] == [{"source": "first"}, None, {"source": "third"}]


@pytest.mark.parametrize("index_type", ["IndexFlatL2", "IndexFlatIP"])
def test_delete_removes_only_requested_chunk_and_ids_remain_unique(embedding, index_type):
    db = FaissVectorDB(embedding, index_type=index_type)
    east, north = chunk("east", 1, 0), chunk("north", 0, 1)
    db.add_chunks([east, north])
    db.delete_chunk(east)
    db.delete_chunk(east)
    db.add_chunk(chunk("west", -1, 0))
    assert db.index.ntotal == 2
    assert set(db.id_map) == {1, 2}
    assert "east" not in db.chunk_id_to_faiss_id
    assert {c.doc_id for c in db.search("east")} == {"north", "west"}


def test_save_load_restores_search_metadata_and_next_id(db, embedding, tmp_path):
    db.add_text("east", {"source": "saved"})
    paths = str(tmp_path / "index.faiss"), str(tmp_path / "ids.pkl")
    db.save(*paths)
    restored = FaissVectorDB(embedding)
    restored.load(*paths)
    result = restored.search("east", k=1)
    assert result[0].metadata == {"source": "saved"}
    assert result[0].doc_id == db.search("east", k=1)[0].doc_id
    restored.add_text("north")
    assert restored.next_id == 2
    assert restored.index.ntotal == 2
    assert len(restored.chunk_id_to_faiss_id) == 2


async def test_all_async_helpers_preserve_arguments_and_results(db):
    await db.aadd_chunk(chunk("first", 1, 0))
    await db.aadd_chunks([chunk("second", 0, 1)])
    await db.aadd_text("west", {"kind": "single"})
    await db.aadd_texts(["east"], [{"kind": "batch"}])
    assert db.index.ntotal == 4
    results = await db.asearch("west", k=1, fetch_k=2)
    assert results[0].content == "west"
    assert results[0].metadata == {"kind": "single"}
    assert db.id_map[3].metadata == {"kind": "batch"}


def test_mmr_handles_empty_search(db):
    assert db.search("east", rerank="mmr") == []


def test_mmr_works_with_generated_embeddings(db):
    db.add_texts(["east", "north", "west"])
    results = db.search("east", rerank="mmr", k=2)
    assert [item.content for item in results] == ["east", "north"]


def test_mmr_balances_relevance_and_diversity(db):
    docs = np.array([[1, 0], [0.9, 0.1], [0, 1]], dtype=np.float32)
    assert db._mmr(vec(1, 0), docs, lambda_=0.3, top_n=2) == [0, 2]
    assert db._mmr(vec(1, 0), docs, lambda_=1, top_n=2) == [0, 1]


def test_mmr_can_select_scores_below_minus_one(db):
    assert db._mmr(vec(2, 0), np.array([[-2, 0]], dtype=np.float32), top_n=2) == [0]


def test_inner_product_ranks_by_dot_product(embedding):
    db = FaissVectorDB(embedding, index_type="IndexFlatIP")
    exact, larger = chunk("exact", 1, 0), chunk("larger", 100, 1)
    db.add_chunks([exact, larger])
    assert db.search("east", k=2) == [larger, exact]


def test_single_generated_vector_may_be_one_dimensional(db, embedding):
    embedding.embed.side_effect = None
    embedding.embed.return_value = vec(1, 0)
    db.add_chunk(chunk("east"))
    assert db.search("east", k=1)[0].content == "east"


def test_multiple_batches_keep_ids_and_generated_embeddings_aligned(db, embedding):
    chunks = [Chunk(content="east") for _ in range(1025)]
    db.add_chunks(chunks)
    assert [len(call.args[0]) for call in embedding.embed.call_args_list] == [1024, 1]
    assert db.index.ntotal == db.next_id == len(db.id_map) == 1025
    assert db.id_map[1024] is chunks[-1]
    assert len(db.chunk_id_to_faiss_id) == 1025
    np.testing.assert_array_equal(chunks[-1].embedding, vec(1, 0))


def test_mmr_keeps_mixed_precomputed_and_generated_results_aligned(db):
    east, north, west = chunk("east"), chunk("north", 0, 1), chunk("west")
    db.add_chunks([north, east, west])
    assert db.search("east", rerank="mmr", k=2) == [east, north]


def test_delete_unknown_chunk_is_noop(db):
    db.add_chunk(chunk("east", 1, 0))
    db.delete_chunk(chunk("missing"))
    assert db.index.ntotal == 1
    assert db.search("east")[0].doc_id == "east"


def test_empty_database_round_trip(db, embedding, tmp_path):
    paths = str(tmp_path / "empty.faiss"), str(tmp_path / "empty.pkl")
    db.save(*paths)
    restored = FaissVectorDB(embedding)
    restored.load(*paths)
    assert restored.search("east") == []
    assert restored.next_id == 0


@pytest.mark.parametrize("index_type", ["IndexFlatL2", "IndexFlatIP"])
def test_delete_migrates_persisted_legacy_hnsw_without_reembedding(embedding, tmp_path, index_type):
    import pickle

    import faiss

    # Recreate the prior on-disk format, including absent chunk embeddings and
    # sparse external IDs that do not equal HNSW's internal row positions.
    inner = faiss.IndexHNSWFlat(2, 32)
    if index_type == "IndexFlatIP":
        inner.metric_type = faiss.METRIC_INNER_PRODUCT
    legacy_index = faiss.IndexIDMap(inner)
    legacy_index.add_with_ids(np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float32), np.array([3, 8, 14], dtype=np.int64))
    chunks = {3: chunk("east"), 8: chunk("north"), 14: chunk("west")}
    paths = str(tmp_path / "legacy.faiss"), str(tmp_path / "legacy.pkl")
    faiss.write_index(legacy_index, paths[0])
    with open(paths[1], "wb") as handle:
        pickle.dump({"id_map": chunks, "chunk_id_to_faiss_id": {item.doc_id: identifier for identifier, item in chunks.items()}, "next_id": 15}, handle)

    db = FaissVectorDB(embedding, index_type=index_type)
    db.load(*paths)
    assert isinstance(faiss.downcast_index(db.index.index), faiss.IndexHNSWFlat)
    db.delete_chunk(chunks[8])
    embedding.embed.assert_not_called()
    assert db.index.ntotal == 2
    assert set(db.id_map) == {3, 14}
    assert db.next_id == 15
    assert db.search("east", k=3) == [chunks[3], chunks[14]]
    assert db.index.metric_type == legacy_index.metric_type

    # The migrated index uses the same persistence format and remains writable.
    db.save(*paths)
    restored = FaissVectorDB(embedding, index_type=index_type)
    restored.load(*paths)
    restored.add_chunk(chunk("new", 0, 1))
    restored.delete_chunk(chunks[3])
    assert set(restored.id_map) == {14, 15}
    assert restored.next_id == 16
    assert {item.doc_id for item in restored.search("east")} == {"west", "new"}


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("index_type", ["IndexFlatL2", "IndexFlatIP"])
def test_load_restores_missing_vectors_for_mmr_without_provider_calls(embedding, tmp_path, legacy, index_type):
    import faiss

    original = FaissVectorDB(embedding, index_type=index_type)
    if legacy:
        inner = faiss.IndexHNSWFlat(2, 32)
        if index_type == "IndexFlatIP":
            inner.metric_type = faiss.METRIC_INNER_PRODUCT
        original.index = faiss.IndexIDMap(inner)
    original.next_id = 7
    items = [chunk("east", 1, 0), chunk("north", 0, 1), chunk("west", -1, 0)]
    original.add_chunks(items)
    # Prior releases left generated chunk embeddings absent in the metadata.
    items[0].embedding = items[2].embedding = None
    paths = str(tmp_path / "vectors.faiss"), str(tmp_path / "vectors.pkl")
    original.save(*paths)

    restored = FaissVectorDB(embedding, index_type=index_type)
    restored.load(*paths)
    embedding.embed.assert_not_called()
    for identifier, expected in [(7, [1, 0]), (8, [0, 1]), (9, [-1, 0])]:
        np.testing.assert_array_equal(restored.id_map[identifier].embedding, expected)
    assert restored.search("east", k=2, rerank="mmr") == [items[0], items[1]]
    embedding.embed.assert_called_once_with("east")
