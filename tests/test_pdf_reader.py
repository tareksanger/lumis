from types import SimpleNamespace
from unittest.mock import Mock

import lumis.reader.pdf as module
from lumis.reader.pdf import PDFReader

import pytest
import requests


@pytest.mark.parametrize("value,expected", [("https://example.com/a.pdf", True), ("http://example.org", True), ("/tmp/a.pdf", False), ("text only", False)])
def test_url_detection(value, expected):
    assert PDFReader.is_url(value) is expected


def test_pdf_header_and_url_cleaning():
    assert PDFReader.clean_url("https://example.com/doc.pdf?token=1#page2") == "https://example.com/doc.pdf"
    assert PDFReader.is_pdf_content(b"%PDF-1.7")
    assert not PDFReader.is_pdf_content(b"html")
    assert PDFReader.is_pdf_content_type({"Content-Type": "application/pdf"})
    assert not PDFReader.is_pdf_content_type({})


@pytest.fixture
def reader(monkeypatch):
    reader = Mock(return_value=SimpleNamespace(pages=[Mock(extract_text=Mock(return_value="first")), Mock(extract_text=Mock(return_value="second"))]))
    monkeypatch.setattr(module.PyPDF2, "PdfReader", reader)
    return reader


def test_local_pdf_extracts_all_pages(tmp_path, reader):
    path = tmp_path / "document.PDF"
    path.write_bytes(b"%PDF-test")
    assert PDFReader.read(str(path)) == "firstsecond"
    assert reader.call_args.args[0].closed


def test_remote_pdf_extracts_pages(monkeypatch, reader):
    response = Mock(headers={"Content-Type": "application/pdf"}, content=b"%PDF-test")
    get = Mock(return_value=response)
    monkeypatch.setattr(module.requests, "get", get)
    assert PDFReader.read("https://example.org/doc.pdf?token=1") == "firstsecond"
    get.assert_called_once_with("https://example.org/doc.pdf?token=1")
    response.raise_for_status.assert_called_once()


@pytest.mark.parametrize("headers,content", [({}, b"%PDF"), ({"Content-Type": "application/pdf"}, b"not-pdf")])
def test_remote_non_pdf_rejected(monkeypatch, headers, content):
    monkeypatch.setattr(module.requests, "get", Mock(return_value=Mock(headers=headers, content=content)))
    with pytest.raises(ValueError, match="PDF"):
        PDFReader.pdf_from_url_to_string("https://example.org/a")


def test_http_error_propagates(monkeypatch):
    response = Mock()
    response.raise_for_status.side_effect = requests.HTTPError("404")
    monkeypatch.setattr(module.requests, "get", Mock(return_value=response))
    with pytest.raises(requests.HTTPError):
        PDFReader.pdf_from_url_to_string("https://example.org/a")


def test_invalid_paths(tmp_path):
    with pytest.raises(ValueError, match="neither"):
        PDFReader.read(str(tmp_path / "missing.pdf"))
    path = tmp_path / "data.txt"
    path.write_text("text")
    with pytest.raises(ValueError, match="not a PDF"):
        PDFReader.read(str(path))
