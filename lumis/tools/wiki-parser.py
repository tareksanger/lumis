# from typing import Any


# class WikiReader:
#     def __init__(self) -> None:
#         """Initialize with parameters."""
#         try:
#             import wikipedia  # noqa
#         except ImportError:
#             raise ImportError(
#                 "`wikipedia` package not found, please run `pip install wikipedia`"
#             )

#     def load_data(
#         self, pages: list[str], lang_prefix: str = "en", **load_kwargs: Any
#     ) -> list[Document]:
#         """Load data from the input directory.

#         Args:
#             pages (List[str]): List of pages to read.
#             lang_prefix (str): Language prefix for Wikipedia. Defaults to English. Valid Wikipedia language codes
#             can be found at https://en.wikipedia.org/wiki/List_of_Wikipedias.
#         """
#         import wikipedia

#         if lang_prefix.lower() != "en":
#             if lang_prefix.lower() in wikipedia.languages():
#                 wikipedia.set_lang(lang_prefix.lower())
#             else:
#                 raise ValueError(
#                     f"Language prefix '{lang_prefix}' for Wikipedia is not supported. Check supported languages at https://en.wikipedia.org/wiki/List_of_Wikipedias."
#                 )
#         value = wikipedia.search("")

#         results = []
#         for page in pages:
#             wiki_page = wikipedia.page(page, **load_kwargs)
#             page_content = wiki_page.content
#             page_id = wiki_page.pageid
#             results.append(Document(id_=page_id, text=page_content))
#         return results
