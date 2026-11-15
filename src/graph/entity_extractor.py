"""
Entity Extractor for GraphRAG.

Extracts entities and relations from paper text using the LLM with robust fallback
handling for CPU-friendly models like qwen2.5-coder:3b.
"""
import json
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from src.core.llm import OllamaLLM, get_llm
from src.graph.schema import (
    Entity, Relation, EntityType, RelationType,
    Entity as EntityClass, Relation as RelationClass
)

logger = logging.getLogger(__name__)


def _normalize_entity_id(name: str) -> str:
    """
    Strongly normalize entity name to a canonical ID.

    Steps:
    1. lowercase
    2. replace spaces and hyphens with single hyphen
    3. strip punctuation (non-alphanumeric)
    4. collapse repeated hyphens
    5. remove leading/trailing hyphens
    6. limit length
    """
    # 1. lowercase
    normalized = name.lower().strip()
    # 2. replace spaces and hyphens with single hyphen
    normalized = re.sub(r'[\s\-]+', '-', normalized)
    # 3. strip punctuation (keep only alphanumeric and hyphens)
    normalized = re.sub(r'[^a-z0-9-]', '', normalized)
    # 4. collapse repeated hyphens
    normalized = re.sub(r'-+', '-', normalized)
    # 5. remove leading/trailing hyphens
    normalized = normalized.strip('-')
    # 6. limit length
    if len(normalized) > 64:
        normalized = normalized[:64].strip('-')
    # 7. if result is empty, fall back to a simplified version
    if not normalized:
        normalized = re.sub(r'[^a-z0-9]', '', name.lower())[:32].strip('-')
    return normalized


def _parse_json_from_response(response: str) -> Optional[Dict[str, Any]]:
    """
    Parse JSON from LLM response with multiple recovery strategies.

    1. Try direct JSON parse
    2. Strip markdown fences and try again
    3. Extract first {...} block
    4. One-pass repair prompt
    """
    response = response.strip()

    # 1. Try direct JSON parse
    try:
        return json.loads(response)
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. Strip markdown code fences
    cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', response, flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        pass

    # 3. Extract first {...} block
    match = re.search(r'\{.*\}', response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except (json.JSONDecodeError, ValueError):
            pass

    # 4. One-pass repair: ask LLM to fix to valid JSON only
    try:
        repair_prompt = """The user asked for JSON output. The previous response was not valid JSON.
Please output ONLY a valid JSON object with "entities" and "relations" keys.
Do not include any reasoning, explanation, or text outside the JSON.

If you cannot extract valid entities/relations, output:
{"entities": [], "relations": []}

Query: {query}
"""
        # Try to extract a query from response context, otherwise use generic
        repair_response = llm_generate(repair_prompt.replace("{query}", "Fix this to valid JSON"))
        # Try parse the repair response
        try:
            return json.loads(repair_response.strip())
        except (json.JSONDecodeError, ValueError):
            pass
    except Exception:
        pass

    return None


def llm_generate(prompt: str, system_prompt: str = "", temperature: float = 0.0,
                 max_tokens: int = 512) -> str:
    """Generate text from LLM with given prompt."""
    from src.core.llm import get_llm
    llm = get_llm()
    return llm.generate(
        prompt=prompt,
        system_prompt=system_prompt if system_prompt else None,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _parse_entities(entities_data: List[Dict[str, Any]], paper_id: str) -> List[Entity]:
    """Parse and normalize entity data from LLM response."""
    entities = []
    for e in entities_data:
        name = e.get("name", "").strip()
        if not name:
            continue
        entity_id = _normalize_entity_id(name)
        entity_type = e.get("type", "Other")
        aliases = e.get("aliases", [])
        if isinstance(aliases, str):
            aliases = [aliases]
        aliases = [a.strip() for a in aliases if a.strip()]

        entity = Entity(
            id=entity_id,
            name=name,
            type=entity_type,
            aliases=aliases,
            paper_ids=[paper_id],
        )
        entities.append(entity)
    return entities


def _parse_relations(
    relations_data: List[Dict[str, Any]],
    paper_id: str,
    chunk_id: str = ""
) -> List[Relation]:
    """Parse and normalize relation data from LLM response."""
    relations = []
    for r in relations_data:
        source = r.get("source", "").strip()
        target = r.get("target", "").strip()
        rel_type = r.get("relation_type", "related_to").strip().lower()
        evidence = r.get("evidence", "").strip()

        if not source or not target or not evidence:
            continue

        source_id = _normalize_entity_id(source)
        target_id = _normalize_entity_id(target)
        rel_id = f"{source_id}_{rel_type}_{target_id}"

        relation = Relation(
            id=rel_id,
            source_id=source_id,
            relation_type=rel_type,
            target_id=target_id,
            evidence=evidence,
            chunk_ids=[chunk_id] if chunk_id else [],
            paper_id=paper_id,
            weight=1.0,
        )
        relations.append(relation)
    return relations


class EntityExtractor:
    """
    Extracts entities and relations from text using LLM with robust fallbacks.

    Uses a staged approach to maximize success rate on CPU-friendly models:
    A) Normal extraction with full text
    B) Shorter text (600 chars) + simpler schema (entities only)
    C) Entities-only from original text
    Never crashes the build.
    """

    # Stage A: Full extraction prompt
    SYSTEM_PROMPT_A = """You are extracting entities and relations from an AI research paper abstract.

Return EXACTLY valid JSON with this structure - nothing else:
{
  "entities": [
    {"name": "Self-RAG", "type": "Method", "aliases": ["Self RAG"]},
    {"name": "reflection tokens", "type": "Technique", "aliases": []}
  ],
  "relations": [
    {"source": "Self-RAG", "relation_type": "introduces", "target": "reflection tokens", "evidence": "Self-RAG introduces reflection tokens for self-correction"}
  ]
}

Constraints:
- Max 4 entities total. Preferred entity types: Method, Model, Technique, Metric, Task, Other
- Max 4 relations total. Preferred relation types: introduces, uses, based_on, compares_with, addresses, evaluates_on, improves
- High precision: only extract clear, explicit mentions
- Include aliases for known variations
- Use canonical names: "Self-RAG", "CRAG", "RAPTOR", "BERT", "GPT", "Transformer"
- Temperature: 0 (deterministic)
"""

    # Stage B: Shorter text, entities only
    SYSTEM_PROMPT_B = """You are extracting entities from AI research paper text.

Return EXACTLY valid JSON with this structure - nothing else:
{
  "entities": [
    {"name": "Self-RAG", "type": "Method", "aliases": ["Self RAG"]},
    {"name": "reflection", "type": "Technique", "aliases": []}
  ]
}

Constraints:
- Max 4 entities total
- Do NOT include "relations" key
- Use these entity types: Method, Model, Technique, Metric, Task, Other
- High precision only
- Temperature: 0 (deterministic)
"""

    # Stage C: Entities-only from text
    SYSTEM_PROMPT_C = """You are extracting entities from this AI research paper text.

Return EXACTLY valid JSON with this structure - nothing else:
{
  "entities": [
    {"name": "Self-RAG", "type": "Method", "aliases": ["Self RAG"]}
  ]
}

Constraints:
- Max 4 entities total
- Do NOT include "relations" key
- Only extract entities clearly mentioned in the text
- Temperature: 0 (deterministic)
"""

    def __init__(
        self,
        llm: Optional[OllamaLLM] = None,
        max_entities: int = 4,
        max_relations: int = 4,
    ):
        """
        Initialize entity extractor.

        Args:
            llm: LLM client (creates default if None)
            max_entities: Maximum entities to extract (default 4 for CPU-friendliness)
            max_relations: Maximum relations to extract (default 4 for CPU-friendliness)
        """
        self.llm = llm or get_llm()
        self.max_entities = max_entities
        self.max_relations = max_relations

    def extract(
        self,
        text: str,
        paper_id: str,
        chunk_id: str = "",
    ) -> Tuple[List[Entity], List[Relation]]:
        """
        Extract entities and relations from text with multi-stage fallback.

        Extraction order:
        A) Normal extraction with full text + entities + relations
        B) If A fails: shorter text (600 chars) + entities only
        C) If B fails: entities only from original text

        Always returns (entities, relations) - never crashes.

        Args:
            text: Text to extract from
            paper_id: Paper identifier
            chunk_id: Optional chunk identifier

        Returns:
            Tuple of (entities, relations) - one will be empty if fallback used
        """
        # Stage A: Normal extraction
        entities, relations = self._stage_a(text, paper_id, chunk_id)
        if entities is not None:
            logger.debug(f"Stage A succeeded: {len(entities)} entities, {len(relations)} relations")
            return entities, relations

        # Stage B: Shorter text, entities only
        logger.debug("Stage A failed, trying Stage B (shorter text, entities only)")
        entities, _ = self._stage_b(text, paper_id, chunk_id)
        if entities is not None:
            logger.debug(f"Stage B succeeded: {len(entities)} entities (relations omitted)")
            return entities, []

        # Stage C: Entities only from original text
        logger.debug("Stage B failed, trying Stage C (entities only from original text)")
        entities, _ = self._stage_c(text, paper_id, chunk_id)
        if entities is not None:
            logger.debug(f"Stage C succeeded: {len(entities)} entities (relations omitted)")
            return entities, []

        # Absolute fallback - never crash
        logger.warning("All extraction stages failed, returning empty results")
        return [], []

    def _stage_a(self, text: str, paper_id: str, chunk_id: str) -> Tuple[Optional[List[Entity]], Optional[List[Relation]]]:
        """Stage A: Full extraction with entities + relations."""
        try:
            # Truncate text to avoid token limits
            text_truncated = text[:900] if len(text) > 900 else text

            prompt = f"""Extract entities and relations from this research paper abstract.

Abstract:
{text_truncated}

Return JSON only:"""

            response = self.llm.generate(
                prompt=prompt,
                system_prompt=self.SYSTEM_PROMPT_A,
                temperature=0.0,
                max_tokens=512,
            )

            data = _parse_json_from_response(response)
            if data is None:
                return None, None

            entities = _parse_entities(data.get("entities", []), paper_id)
            relations = _parse_relations(data.get("relations", []), paper_id, chunk_id)

            # Apply caps
            entities = entities[:self.max_entities]
            relations = relations[:self.max_relations]

            return entities, relations

        except Exception as e:
            logger.warning(f"Stage A failed: {e}")
            return None, None

    def _stage_b(self, text: str, paper_id: str, chunk_id: str) -> Tuple[Optional[List[Entity]], None]:
        """Stage B: Shorter text (600 chars), entities only."""
        try:
            text_truncated = text[:600] if len(text) > 600 else text

            prompt = f"""Extract entities from this research paper abstract.

Abstract:
{text_truncated}

Return JSON only:"""

            response = self.llm.generate(
                prompt=prompt,
                system_prompt=self.SYSTEM_PROMPT_B,
                temperature=0.0,
                max_tokens=512,
            )

            data = _parse_json_from_response(response)
            if data is None:
                return None, None

            entities = _parse_entities(data.get("entities", []), paper_id)

            return entities, None

        except Exception as e:
            logger.warning(f"Stage B failed: {e}")
            return None, None

    def _stage_c(self, text: str, paper_id: str, chunk_id: str) -> Tuple[Optional[List[Entity]], None]:
        """Stage C: Entities only from original text."""
        try:
            prompt = f"""Extract entities from this research paper text.

Text:
{text}

Return JSON only:"""

            response = self.llm.generate(
                prompt=prompt,
                system_prompt=self.SYSTEM_PROMPT_C,
                temperature=0.0,
                max_tokens=512,
            )

            data = _parse_json_from_response(response)
            if data is None:
                return None, None

            entities = _parse_entities(data.get("entities", []), paper_id)

            return entities, None

        except Exception as e:
            logger.warning(f"Stage C failed: {e}")
            return None, None


def get_entity_extractor(
    llm: Optional[OllamaLLM] = None,
    max_entities: int = 4,
    max_relations: int = 4,
) -> EntityExtractor:
    """Get an entity extractor instance."""
    return EntityExtractor(llm, max_entities, max_relations)