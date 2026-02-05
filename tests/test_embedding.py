from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

from lumis.embedding.base_embedding import BaseEmbeddingModel

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Concrete stub for the abstract base
# ---------------------------------------------------------------------------


class _StubEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, dimension: int = 3):
        super().__init__("stub-model")
        self._dimension = dimension

    def embed(self, text):
        raise NotImplementedError

    async def aembed(self, text):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# BaseEmbeddingModel – dimension property
# ---------------------------------------------------------------------------


class TestBaseEmbeddingDimension:
    def test_returns_configured_dimension(self):
        assert _StubEmbeddingModel(dimension=128).dimension == 128

    def test_raises_when_dimension_is_zero(self):
        model = _StubEmbeddingModel(dimension=0)
        with pytest.raises(AssertionError, match="Dimension not set"):
            _ = model.dimension


# ---------------------------------------------------------------------------
# BaseEmbeddingModel – similarity
# ---------------------------------------------------------------------------


class TestBaseEmbeddingSimilarity:
    @pytest.fixture
    def model(self):
        return _StubEmbeddingModel()

    # -- cosine --

    def test_cosine_identical_vectors(self, model):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert model.similarity(v, v, mode="cosine") == pytest.approx(1.0)

    def test_cosine_orthogonal_vectors(self, model):
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        assert model.similarity(v1, v2, mode="cosine") == pytest.approx(0.0)

    def test_cosine_opposite_vectors(self, model):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert model.similarity(v, -v, mode="cosine") == pytest.approx(-1.0)

    def test_cosine_is_default_mode(self, model):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert model.similarity(v, v) == pytest.approx(1.0)

    # -- dot product --

    def test_dot_product(self, model):
        v1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        v2 = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        assert model.similarity(v1, v2, mode="dot_product") == pytest.approx(float(np.dot(v1, v2)))

    # -- euclidean --

    def test_euclidean_identical_vectors(self, model):
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert model.similarity(v, v, mode="euclidean") == pytest.approx(0.0)

    def test_euclidean_known_distance(self, model):
        # 3-4-5 triangle → distance = 5 → similarity = -5
        v1 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([3.0, 4.0, 0.0], dtype=np.float32)
        assert model.similarity(v1, v2, mode="euclidean") == pytest.approx(-5.0)


# ---------------------------------------------------------------------------
# OpenAIEmbeddingModel
# ---------------------------------------------------------------------------


class TestOpenAIEmbeddingModel:
    @pytest.fixture
    def model(self):
        with patch("lumis.embedding.openai_embedding.OpenAI"), patch(
            "lumis.embedding.openai_embedding.AsyncOpenAI"
        ) as MockAsync:
            from lumis.embedding.openai_embedding import OpenAIEmbeddingModel

            m = OpenAIEmbeddingModel(model_name="text-embedding-3-small")
            m.aclient = MockAsync.return_value
            return m

    # -- dimension resolution --

    def test_dimension_from_known_model(self):
        with patch("lumis.embedding.openai_embedding.OpenAI"), patch(
            "lumis.embedding.openai_embedding.AsyncOpenAI"
        ):
            from lumis.embedding.openai_embedding import OpenAIEmbeddingModel

            assert OpenAIEmbeddingModel("text-embedding-3-large").dimension == 3072

    def test_dimension_custom_override(self):
        with patch("lumis.embedding.openai_embedding.OpenAI"), patch(
            "lumis.embedding.openai_embedding.AsyncOpenAI"
        ):
            from lumis.embedding.openai_embedding import OpenAIEmbeddingModel

            assert OpenAIEmbeddingModel("text-embedding-3-large", dimension=512).dimension == 512

    # -- aembed edge-cases (no network call) --

    async def test_aembed_empty_string_returns_zero_vector(self, model):
        result = await model.aembed("")
        assert result.shape == (model.dimension,)
        np.testing.assert_array_equal(result, np.zeros(model.dimension, dtype="float32"))

    async def test_aembed_empty_list_returns_empty_array(self, model):
        result = await model.aembed([])
        assert result.shape == (0, model.dimension)

    # -- aembed with mocked API --

    async def test_aembed_single_string(self, model):
        dim = model.dimension
        resp = MagicMock()
        resp.data = [MagicMock(embedding=[0.1] * dim)]
        model.aclient.embeddings.create = AsyncMock(return_value=resp)

        result = await model.aembed("hello")

        assert result.shape == (dim,)
        model.aclient.embeddings.create.assert_awaited_once_with(
            model="text-embedding-3-small", input=["hello"]
        )

    async def test_aembed_list_of_strings(self, model):
        dim = model.dimension
        resp = MagicMock()
        resp.data = [MagicMock(embedding=[0.1] * dim), MagicMock(embedding=[0.2] * dim)]
        model.aclient.embeddings.create = AsyncMock(return_value=resp)

        result = await model.aembed(["hello", "world"])

        assert result.shape == (2, dim)


# ---------------------------------------------------------------------------
# HuggingFaceEmbeddingModel
# ---------------------------------------------------------------------------


class TestHuggingFaceEmbeddingModel:
    @pytest.fixture
    def model(self):
        mock_st_instance = MagicMock()
        mock_st_instance.get_sentence_embedding_dimension.return_value = 384

        mock_module = MagicMock()
        mock_module.SentenceTransformer = MagicMock(return_value=mock_st_instance)

        with patch.dict(sys.modules, {"sentence_transformers": mock_module}):
            from lumis.embedding.huggingface_embedding import HuggingFaceEmbeddingModel

            return HuggingFaceEmbeddingModel()

    def test_dimension(self, model):
        assert model.dimension == 384

    def test_embed_wraps_single_string_in_list(self, model):
        model.model.encode.return_value = np.array([[0.1] * 384])

        result = model.embed("hello")

        model.model.encode.assert_called_once_with(["hello"], convert_to_numpy=True)
        assert result.shape == (1, 384)

    def test_embed_passes_list_through(self, model):
        model.model.encode.return_value = np.array([[0.1] * 384, [0.2] * 384])

        result = model.embed(["hello", "world"])

        model.model.encode.assert_called_once_with(["hello", "world"], convert_to_numpy=True)
        assert result.shape == (2, 384)

    async def test_aembed_wraps_single_string(self, model):
        model.model.encode.return_value = np.array([[0.1] * 384])

        result = await model.aembed("hello")

        assert result.shape == (1, 384)

    def test_import_error_when_sentence_transformers_missing(self):
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            from lumis.embedding.huggingface_embedding import HuggingFaceEmbeddingModel

            with pytest.raises(ImportError, match="sentence_transformers"):
                HuggingFaceEmbeddingModel()
