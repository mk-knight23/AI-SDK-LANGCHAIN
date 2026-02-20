"""Tests for EnterpriseRAG implementation."""
import pytest
from unittest.mock import Mock, patch, MagicMock


class TestEnterpriseRAG:
    """Test suite for EnterpriseRAG class."""

    @pytest.fixture
    def mock_chroma_client(self):
        """Create a mock ChromaDB client."""
        with patch('chromadb.Client') as mock_client:
            mock_collection = Mock()
            mock_collection.add = Mock()
            mock_collection.query = Mock(return_value={
                'documents': [['Test document content']],
                'metadatas': [[{'source': 'test'}]],
                'distances': [[0.1]]
            })
            mock_client.return_value.get_or_create_collection.return_value = mock_collection
            yield mock_client, mock_collection

    @pytest.fixture
    def rag_instance(self, mock_chroma_client):
        """Create an EnterpriseRAG instance with mocked ChromaDB."""
        mock_client, mock_collection = mock_chroma_client
        with patch('app.rag.ChromaDB') as mock_chroma_class:
            mock_chroma_instance = Mock()
            mock_chroma_instance.client = mock_client.return_value
            mock_chroma_instance.collection = mock_collection
            mock_chroma_class.return_value = mock_chroma_instance

            from app.rag import EnterpriseRAG
            rag = EnterpriseRAG(collection_name="test_collection")
            rag.collection = mock_collection
            return rag

    def test_add_documents_success(self, rag_instance):
        """Test adding documents successfully."""
        documents = [
            {"content": "Document 1", "metadata": {"source": "doc1"}},
            {"content": "Document 2", "metadata": {"source": "doc2"}}
        ]

        rag_instance.add_documents(documents)

        rag_instance.collection.add.assert_called_once()
        call_args = rag_instance.collection.add.call_args

        assert call_args.kwargs['documents'] == ["Document 1", "Document 2"]
        assert call_args.kwargs['metadatas'] == [{"source": "doc1"}, {"source": "doc2"}]
        assert len(call_args.kwargs['ids']) == 2

    def test_add_documents_empty_list(self, rag_instance):
        """Test handling empty document list."""
        rag_instance.add_documents([])

        rag_instance.collection.add.assert_not_called()

    def test_add_documents_with_ids(self, rag_instance):
        """Test adding documents with custom IDs."""
        documents = [
            {"content": "Document 1", "metadata": {"source": "doc1"}, "id": "custom_id_1"}
        ]

        rag_instance.add_documents(documents)

        call_args = rag_instance.collection.add.call_args
        assert call_args.kwargs['ids'] == ["custom_id_1"]

    def test_query_success(self, rag_instance):
        """Test querying documents successfully."""
        rag_instance.collection.query.return_value = {
            'documents': [['Relevant document content']],
            'metadatas': [[{'source': 'test_doc'}]],
            'distances': [[0.25]]
        }

        results = rag_instance.query("test query", n_results=3)

        rag_instance.collection.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=3
        )
        assert len(results) == 1
        assert results[0]['content'] == 'Relevant document content'
        assert results[0]['metadata'] == {'source': 'test_doc'}
        assert results[0]['distance'] == 0.25

    def test_query_empty_results(self, rag_instance):
        """Test querying with no results."""
        rag_instance.collection.query.return_value = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }

        results = rag_instance.query("nonexistent query")

        assert results == []

    def test_query_default_n_results(self, rag_instance):
        """Test query uses default n_results value."""
        rag_instance.collection.query.return_value = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }

        rag_instance.query("test query")

        call_args = rag_instance.collection.query.call_args
        assert call_args.kwargs['n_results'] == 5

    def test_add_documents_generates_unique_ids(self, rag_instance):
        """Test that unique IDs are generated when not provided."""
        documents = [
            {"content": "Document 1", "metadata": {}},
            {"content": "Document 2", "metadata": {}}
        ]

        rag_instance.add_documents(documents)

        call_args = rag_instance.collection.add.call_args
        ids = call_args.kwargs['ids']
        assert len(ids) == 2
        assert ids[0] != ids[1]
        assert all(isinstance(id_, str) for id_ in ids)

    @patch('chromadb.Client')
    def test_chroma_db_initialization(self, mock_client):
        """Test ChromaDB client is initialized correctly."""
        mock_collection = Mock()
        mock_client.return_value.get_or_create_collection.return_value = mock_collection

        with patch('app.rag.ChromaDB') as mock_chroma_class:
            mock_chroma_instance = Mock()
            mock_chroma_instance.client = mock_client.return_value
            mock_chroma_instance.collection = mock_collection
            mock_chroma_class.return_value = mock_chroma_instance

            from app.rag import EnterpriseRAG
            rag = EnterpriseRAG(collection_name="my_collection")

            assert rag.collection_name == "my_collection"

    def test_query_with_filter(self, rag_instance):
        """Test querying with metadata filter."""
        rag_instance.collection.query.return_value = {
            'documents': [['Filtered result']],
            'metadatas': [[{'source': 'filtered'}]],
            'distances': [[0.1]]
        }

        results = rag_instance.query(
            "test query",
            n_results=2,
            filter={"source": "doc1"}
        )

        call_args = rag_instance.collection.query.call_args
        assert call_args.kwargs['where'] == {"source": "doc1"}
