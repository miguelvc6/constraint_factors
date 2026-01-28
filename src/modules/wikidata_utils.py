import logging
import os
import time
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Sequence, TypeAlias, Union
from urllib.parse import urlparse

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from torch import Tensor
from tqdm import tqdm
from urllib3.util.retry import Retry

# Hugging Face Transformers defaults to importing TensorFlow/Keras when available,
# but this project only uses the PyTorch stack. Force TF off (and PyTorch on) so
# we avoid the unsupported tf-keras dependency chain when importing transformers.
os.environ["USE_TF"] = "0"
os.environ.setdefault("USE_TORCH", "1")

from sentence_transformers import SentenceTransformer

Embedding: TypeAlias = Union[np.ndarray, Tensor]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _LookupSpec:
    """Captures how to look up a Wikidata label for an input URI."""

    original: str
    cleaned: str
    entity_id: str
    label_suffix: str = ""

    @property
    def canonical_uri(self) -> str:
        return f"https://www.wikidata.org/entity/{self.entity_id}"


def _normalise_uri_for_lookup(uri: str | None) -> _LookupSpec | None:
    if uri is None:
        return None

    trimmed = uri.strip()
    if not trimmed:
        return None

    cleaned = trimmed.strip("<>").strip()
    if not cleaned:
        return None

    while cleaned.startswith("^"):
        cleaned = cleaned[1:].lstrip()

    label_suffix = ""
    entity_id: str | None = None
    cleaned_for_lookup = cleaned

    if cleaned.startswith("wd:"):
        entity_id = cleaned.split(":", 1)[1]
        cleaned_for_lookup = f"https://www.wikidata.org/entity/{entity_id}"
    elif cleaned.startswith("http://") or cleaned.startswith("https://"):
        parsed = urlparse(cleaned)
        if not parsed.netloc and parsed.path.startswith("www.wikidata.org"):
            parsed = urlparse(f"https://{cleaned}")

        if "wikidata.org" not in parsed.netloc:
            return None

        path = (parsed.path or "").replace("//", "/")
        if path.endswith("/") and len(path) > 1:
            path = path.rstrip("/")
        if path.startswith("/wiki/"):
            entity_id = path.split("/wiki/", 1)[1]
        elif path.startswith("/entity/"):
            entity_id = path.split("/entity/", 1)[1]
        else:
            return None

        entity_id = entity_id.strip().strip("/")
        scheme = parsed.scheme or "https"
        cleaned_for_lookup = f"{scheme}://{parsed.netloc}{path}"
    else:
        entity_id = cleaned.strip()
        if not entity_id:
            return None
        cleaned_for_lookup = f"https://www.wikidata.org/entity/{entity_id}"

    if entity_id is None:
        return None

    if entity_id.startswith("statement/"):
        entity_id = entity_id.split("/", 1)[1] if "/" in entity_id else entity_id[len("statement/") :]
        label_suffix = " (statement)"

    if "-" in entity_id and entity_id[0].upper() in {"P", "Q"}:
        base, remainder = entity_id.split("-", 1)
        if base:
            entity_id = base
            label_suffix = label_suffix or " (statement)"

    entity_id = entity_id.strip()
    if not entity_id:
        return None

    prefix = entity_id[0].upper()
    if prefix not in {"P", "Q", "L", "E"}:
        return None

    return _LookupSpec(
        original=uri,
        cleaned=cleaned_for_lookup,
        entity_id=entity_id,
        label_suffix=label_suffix,
    )


def _looks_like_literal(value: str | None) -> bool:
    if value is None:
        return True
    trimmed = value.strip()
    if not trimmed:
        return True
    if "^^" in trimmed:
        return True
    if trimmed.count('"') >= 2 or (trimmed.startswith('"') and trimmed.endswith('"')):
        return True
    return False


def _build_requests_session() -> requests.Session:
    retry = Retry(
        total=3,
        backoff_factor=1.5,
        status_forcelist=[403, 429, 500, 502, 503, 504],
        allowed_methods={"GET", "POST"},
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _chunk_ids(entity_ids: Sequence[str], chunk_size: int) -> Iterable[list[str]]:
    chunk_size = max(1, chunk_size)
    for index in range(0, len(entity_ids), chunk_size):
        yield list(entity_ids[index : index + chunk_size])


def _query_wikidata_labels(
    session: requests.Session,
    entity_ids: Sequence[str],
    language: str,
) -> list[dict]:
    if not entity_ids:
        return []

    endpoint = "https://query.wikidata.org/sparql"
    # IDENTIFICATION: Use the exact header that worked in your curl test
    headers = {
        "User-Agent": "ConstraintFactorBot/1.0 (https://github.com/your-repo; miguel.vazquez@wu.ac.at) Python-Requests/2.31",
        "Accept": "application/sparql-results+json",
    }

    pending: list[list[str]] = [list(entity_ids)]
    results: list[dict] = []

    # Backoff configuration
    base_delay = 2.0
    max_delay = 60.0

    while pending:
        current = pending.pop()
        if not current:
            continue

        values_clause = " ".join(f"wd:{eid}" for eid in current)
        sparql_query = f"""
            SELECT ?entity ?entityLabel WHERE {{
                VALUES ?entity {{ {values_clause} }}
                SERVICE wikibase:label {{ bd:serviceParam wikibase:language \"{language}\" . }}
            }}
        """

        attempt = 0
        success = False
        payload: dict | None = None

        while not success and attempt < 5:
            try:
                response = session.post(
                    endpoint,
                    data={"query": sparql_query, "format": "json"},
                    headers=headers,
                    timeout=60,
                )

                # 1. HANDLE POLICY BLOCKS IMMEDIATELY
                if response.status_code == 403:
                    logger.error("403 Forbidden: Blocked by Robot Policy. STOPPING to prevent IP ban.")
                    logger.error(f"Response: {response.text[:500]}")
                    return results  # Return progress to avoid data loss

                # 2. HANDLE RATE LIMITS WITH BACKOFF
                if response.status_code == 429:
                    attempt += 1
                    delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                    logger.warning("429 Too Many Requests. Backing off for %.2f seconds...", delay)
                    time.sleep(delay)
                    continue

                response.raise_for_status()
                payload = response.json()
                success = True

                # Be a good citizen: small delay between successful batches
                time.sleep(1.5)

            except requests.Timeout:
                if len(current) > 1:
                    mid = max(1, len(current) // 2)
                    logger.warning("Timeout for batch of %d; splitting.", len(current))
                    pending.append(current[mid:])
                    pending.append(current[:mid])
                    success = True  # Break inner retry to try smaller halves
                else:
                    logger.error("Persistent timeout for ID %s. Skipping.", current[0])
                    break
            except Exception as exc:
                logger.error("Request error for IDs %s: %s", current, exc)
                break

        if success and payload is not None:
            bindings = payload.get("results", {}).get("bindings", [])
            results.extend(bindings)

    return results


def get_wikidata_entity_labels(entity_uris: list[str], language: str = "en") -> dict[str, str]:
    """
    Get labels for Wikidata entities using SPARQL query

    Args:
        entity_uris: list of Wikidata URIs (e.g., ['http://www.wikidata.org/entity/Q123', ...])
        language: Language code for labels (default: 'en')

    Returns:
        dictionary mapping URIs to their labels
    """
    lookup_specs: list[_LookupSpec] = []
    for uri in entity_uris:
        spec = _normalise_uri_for_lookup(uri)
        if spec is not None:
            lookup_specs.append(spec)

    if not lookup_specs:
        return {}

    entity_lookup: dict[str, list[_LookupSpec]] = defaultdict(list)
    for spec in lookup_specs:
        entity_lookup[spec.entity_id].append(spec)

    unique_ids = sorted(entity_lookup.keys())
    session = _build_requests_session()

    labels: dict[str, str] = {}
    for chunk in _chunk_ids(unique_ids, 150):
        bindings = _query_wikidata_labels(session, chunk, language)
        for binding in bindings:
            entity_uri = binding["entity"]["value"]
            entity_id = entity_uri.rsplit("/", 1)[-1]
            label_value = binding.get("entityLabel", {}).get("value", entity_id)
            for spec in entity_lookup.get(entity_id, []):
                label_text = f"{label_value}{spec.label_suffix}" if spec.label_suffix else label_value
                labels.setdefault(spec.canonical_uri, label_value)
                labels.setdefault(f"<{spec.canonical_uri}>", label_value)
                labels[spec.cleaned] = label_text
                labels[spec.original] = label_text
                cleaned_without_brackets = spec.cleaned.strip("<>")
                if cleaned_without_brackets:
                    labels.setdefault(f"<{cleaned_without_brackets}>", label_text)

    return labels


class WikidataUriEmbedder:
    """Resolve Wikidata URIs to labels and compute their embeddings.

    Parameters
    ----------
    batch_size - int:
        Maximum number of URIs to request from the Wikidata endpoint at once.
    big_model - bool:
        Choose a bigger and slower embedding model.
    """

    def __init__(self, batch_size: int = 512, embed_dim: int = 256):
        self._model_name: str = os.getenv("WIKIDATA_EMBEDDING_MODEL", "jinaai/jina-embeddings-v3")
        self._fallback_model_name: str = os.getenv(
            "WIKIDATA_EMBEDDING_FALLBACK",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        if embed_dim in [32, 64, 128, 256, 512, 768, 1024]:
            self.embed_dim = embed_dim
        else:
            raise ValueError(f"Invalid embed_dim {embed_dim}. Choose from [32, 64, 128, 256, 512, 768, 1024].")
        self._sentence_transformer: SentenceTransformer | None = None
        self.batch_size = batch_size
        self.uri2text: dict[str, str] = {}
        self.embeddings_map: dict[str, Embedding] = {}

    def _ensure_model(self) -> SentenceTransformer:
        """Load the embedding model on demand."""
        if self._sentence_transformer is None:
            try:
                self._sentence_transformer = SentenceTransformer(
                    self._model_name,
                    trust_remote_code=True,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to load embedding model %s (%s). Falling back to %s.",
                    self._model_name,
                    exc,
                    self._fallback_model_name,
                )
                self._sentence_transformer = SentenceTransformer(
                    self._fallback_model_name,
                    trust_remote_code=False,
                )
        return self._sentence_transformer

    def _encode_cls(self, texts: list[str] | str):
        model = self._ensure_model()
        return model.encode(
            texts,
            truncate_dim=self.embed_dim,
            show_progress_bar=False,
            task="classification",
        )

    def cache_uri_texts(self, uris: Iterable[str]) -> dict[str, str]:
        unique_uris = set(uris) - set(self.uri2text.keys())

        resolvable: list[str] = []
        for uri in unique_uris:
            if _looks_like_literal(uri):
                self.uri2text.setdefault(uri, uri)
                continue
            if "wikidata.org" not in uri:
                self.uri2text.setdefault(uri, uri)
                continue
            if _normalise_uri_for_lookup(uri) is None:
                self.uri2text.setdefault(uri, uri)
                continue
            resolvable.append(uri)

        uris = resolvable

        # Calculate estimated memory usage
        if uris:
            existing = len(self.uri2text)
            total_expected = existing + len(uris)
            dtype_bytes = np.dtype(np.float16).itemsize
            logging.info(
                "Estimated URI embedding cache size: %.3f GB (count=%d, dim=%d, bytes/elem=%d, dtype=fp16)",
                (total_expected * self.embed_dim * dtype_bytes) / (1024**3),
                total_expected,
                self.embed_dim,
                dtype_bytes,
            )

        # Retrieve labels
        logging.info("Resolving %s URIs to text labels", len(uris))
        for start in tqdm(range(0, len(uris), self.batch_size)):
            batch = uris[start : start + self.batch_size]
            self.uri2text.update(get_wikidata_entity_labels(batch))

        if uris:
            logging.info("Loaded text for %s URIs (total cached: %s)", len(uris), len(self.uri2text))
        return self.uri2text

    def embed_uris(self, uris: Sequence[str]) -> dict[str, Embedding]:
        """Embed a list of uris and returns a dict with uris and the embeddings"""
        self.cache_uri_texts(uris)

        ensured_texts: dict[str, str] = {}
        missed_uris: list[str] = [uri for uri in uris if uri not in self.uri2text]
        if missed_uris:
            logging.warning(
                "Falling back to literal text embedding for %s unresolved URIs (e.g. %s)",
                len(missed_uris),
                missed_uris[:3],
            )
            for uri in missed_uris:
                text = self.get_text(uri)
                self.uri2text.setdefault(uri, text)

        for uri in uris:
            if uri in self.uri2text:
                ensured_texts[uri] = self.uri2text[uri]

        still_missing = [uri for uri in uris if uri not in ensured_texts]
        if still_missing:
            logging.error("Unable to derive text for %s URIs; embedding their raw identifiers", len(still_missing))
            ensured_texts.update({uri: uri for uri in still_missing})

        pending_uris = [uri for uri in uris if uri not in self.embeddings_map]
        if pending_uris:
            embeddings = self._embed_labels({uri: ensured_texts[uri] for uri in pending_uris}, [])
            self.embeddings_map.update(embeddings)

        return {uri: self.embeddings_map[uri] for uri in uris}

    def _embed_labels(self, uri_mapping: dict[str, str], missed_uris: Sequence[str]) -> dict[str, Embedding]:
        """Return embeddings for resolved *uri_mapping* and *missed_uris*."""

        # Embed successful label look-ups
        uris, labels = zip(*uri_mapping.items()) if uri_mapping else ([], [])
        uri_embeddings: np.ndarray = self._encode_cls(list(labels))
        final_embeddings: dict[str, Embedding] = {uri: embedding for uri, embedding in zip(uris, uri_embeddings)}

        # Embed URIs we could not resolve - these are usually literals
        missed_embeddings: np.ndarray = self._encode_cls(list(missed_uris))
        final_embeddings.update({uri: embedding for uri, embedding in zip(missed_uris, missed_embeddings)})
        return final_embeddings

    def get_text(self, uri: str | None) -> str:
        """Return the human-readable label for *uri*."""
        if uri is None:
            raise Exception("Input uri is None")
        if uri not in self.uri2text:
            if "^^" in uri or uri.count('"') >= 2:
                # assume the uri is a literal, e.g. "some text"^^<http://www.w3.org/2001/XMLSchema#string>, or "1234"/"5678"
                return uri
            if "wikidata.org" not in uri:
                # we assume uri is a literal
                logger.debug("URI %s does not contain 'wikidata.org', treating it as a literal.", uri)
                return uri
            spec = _normalise_uri_for_lookup(uri)
            if spec is None:
                logger.debug("URI %s could not be normalised, treating it as a literal.", uri)
                return uri

            wikidata_text = get_wikidata_entity_labels([spec.cleaned])
            label = (
                wikidata_text.get(spec.original)
                or wikidata_text.get(spec.cleaned)
                or wikidata_text.get(spec.canonical_uri)
            )
            if label is None:
                logger.warning("URI %s not found as wikidata entity, using it as a literal.", spec.cleaned)
                return uri
            return label
        return self.uri2text[uri]

    def get_embedding_from_uri(self, uri: str | None) -> Embedding:
        if uri is None:
            raise Exception("Input uri is None")
        if uri not in self.uri2text:
            self.uri2text[uri] = self.get_text(uri)
        text = self.uri2text[uri]
        if uri not in self.embeddings_map:
            self.embeddings_map[uri] = self._ensure_model().encode(
                text,
                truncate_dim=self.embed_dim,
                show_progress_bar=False,
                task="classification",
            )
        return self.embeddings_map[uri]

    def get_embedding_from_text(self, text: str):
        return self._encode_cls(text)

    def get_embeddings_from_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Batch encode literal texts when GPU usage matters."""
        if not texts:
            return np.empty((0, self.embed_dim), dtype=np.float32)
        return np.asarray(self._encode_cls(list(texts)))
