import asyncio
import logging
import threading
from typing import Literal, Optional

import spacy
from spacy.language import Language

logger = logging.getLogger(__name__)

# This is the list of spacy models that can be loaded
SpacyModels = Literal[
    "en_core_web_lg",
    #   "en_core_web_trf
]


class ModelManager(object):
    """
    A singleton class for managing and loading NLP models globally.

    This class ensures that only one instance of the NLP and any other model is loaded
    and shared across the application, optimizing memory usage and
    providing a centralized point for model access.

    Attributes:
        __nlp (Language | None): The loaded spaCy language model.

    Usage:
        model_manager = ModelManager.get_instance()
        await model_manager.load()
        nlp = model_manager.nlp
    """

    _lock = threading.Lock()
    _instance: Optional["ModelManager"] = None

    models: list[SpacyModels] = [
        "en_core_web_lg",
        # "en_core_web_trf"
    ]

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self.__loaded = False
        self.__nlp: dict[SpacyModels, Optional[Language]] = {model: None for model in self.models}

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def nlp(self) -> dict[SpacyModels, Optional[Language]]:
        if not self.__loaded:
            raise ValueError("NLP model not loaded")
        return self.__nlp

    async def load(self):
        """
        Asynchronously loads the specified SpaCy model.

        Args:
            model_name (str): The name of the SpaCy model to load.

        Raises:
            RuntimeError: If the model fails to load.
        """

        for model in self.models:
            if self.__nlp[model] is None:
                try:
                    loop = asyncio.get_event_loop()
                    self.__nlp[model] = await loop.run_in_executor(None, spacy.load, model)
                    logger.debug("")
                except OSError as e:
                    raise RuntimeError(f"Failed to load NLP model '{model}': {e}")

            self.__loaded = True

    def get_spacy_model(self, model_name: SpacyModels):
        model = self.nlp[model_name]
        if model is None:
            raise ValueError(f"Model '{model_name}' found none. Models were not loaded correctly. run 'poetry run python -m spacy download {model_name}'")
        return model
