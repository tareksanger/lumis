from io import BytesIO
import os
import re
from urllib.parse import urlparse, urlunparse

import PyPDF2
import requests


class PDFReader:
    @classmethod
    def is_url(cls, string: str) -> bool:
        """
        Check if a given string is a valid URL.

        Args:
            string (str): The input string to check.

        Returns:
            bool: True if the string is a URL, False otherwise.
        """
        url_regex = re.compile(r"^(https?|ftp)://[^\s/$.?#].[^\s]*$", re.IGNORECASE)
        return re.match(url_regex, string) is not None

    @classmethod
    def clean_url(cls, pdf_url: str) -> str:
        """
        Remove query parameters from the URL.

        Args:
            pdf_url (str): The original URL with or without query parameters.

        Returns:
            str: The URL without query parameters.
        """
        parsed_url = urlparse(pdf_url)
        clean_url = urlunparse((parsed_url.scheme, parsed_url.netloc, parsed_url.path, "", "", ""))
        return clean_url

    @classmethod
    def is_pdf_content_type(cls, headers: dict) -> bool:
        """
        Check if the Content-Type in the response headers is 'application/pdf'.

        Args:
            headers (dict): The HTTP response headers.

        Returns:
            bool: True if Content-Type is 'application/pdf', False otherwise.
        """
        return headers.get("Content-Type") == "application/pdf"

    @classmethod
    def is_pdf_content(cls, content: bytes) -> bool:
        """
        Check if the given content is a valid PDF by examining its header.

        Args:
            content (bytes): The content to check.

        Returns:
            bool: True if the content starts with the PDF header '%PDF', False otherwise.
        """
        return content[:4] == b"%PDF"

    @classmethod
    def pdf_to_string_from_local(cls, file_path: str) -> str:
        """
        Extract text from a local PDF file.

        Args:
            file_path (str): The path to the local PDF file.

        Returns:
            str: The extracted text from the PDF file.

        Raises:
            ValueError: If the file is not a valid PDF.
        """
        with open(file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
            return text

    @classmethod
    def pdf_from_url_to_string(cls, pdf_url: str) -> str:
        """
        Download and extract text from a PDF file at a given URL.

        Args:
            pdf_url (str): The URL pointing to the PDF file.

        Returns:
            str: The extracted text from the PDF file.

        Raises:
            ValueError: If the URL does not lead to a valid PDF.
        """
        response = requests.get(pdf_url)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Check if the response has a PDF Content-Type
        if cls.is_pdf_content_type(response.headers):  # type: ignore
            if cls.is_pdf_content(response.content):
                with BytesIO(response.content) as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    text = ""
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        text += page.extract_text()
                    return text
            else:
                raise ValueError("The content does not appear to be a valid PDF.")
        else:
            raise ValueError("The URL does not return a valid PDF (Content-Type is not 'application/pdf').")

    @classmethod
    def read(cls, path_or_url: str) -> str:
        """
        Dynamically process a string to determine whether it's a URL or a local file path,
        then extract and return the text from the PDF file.

        Args:
            path_or_url (str): The URL or file path pointing to a PDF.

        Returns:
            str: The extracted text from the PDF file.

        Raises:
            ValueError: If the input is neither a valid URL nor a local PDF file.
        """
        if cls.is_url(path_or_url):
            # It's a URL, download and process
            return cls.pdf_from_url_to_string(path_or_url)
        elif os.path.isfile(path_or_url):
            # It's a local file, check if it's a PDF
            if path_or_url.lower().endswith(".pdf"):
                return cls.pdf_to_string_from_local(path_or_url)
            else:
                raise ValueError("The provided local path is not a PDF.")
        else:
            raise ValueError("The input is neither a valid URL nor a local file path.")
