"""
Smoke test for graph provenance and edge constraints.

Verifies:
- Relations with missing endpoints are not stored
- Relations from different papers with same entities are both kept (different IDs)
- Valid relations with both endpoints are stored and retrievable
"""
import pytest
from src.graph.graph_store import GraphStore
from src.graph.schema import Entity, Relation, EntityType, RelationType


class TestGraphProvenance:
    """Tests for graph edge constraints and relation provenance."""

    def setup_method(self):
        """Set up a fresh graph store for each test."""
        import tempfile
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = GraphStore(graph_dir=self.tempdir.name)

    def teardown_method(self):
        """Clean up temp directory."""
        self.tempdir.cleanup()

    def test_relation_missing_source_endpoint_not_stored(self):
        """Relation with missing source entity should be rejected."""
        # Add target entity only
        target = Entity(
            id="target-entity",
            name="Target Entity",
            type=EntityType.METHOD,
            paper_ids=["paper1"],
        )
        self.store.add_entities([target])

        # Try to add relation with missing source
        relation = Relation(
            id="missing-source_relates_to_target-entity_paper1",
            source_id="missing-source",
            relation_type=RelationType.INTRODUCES,
            target_id="target-entity",
            evidence="Test evidence",
            paper_id="paper1",
        )

        count = self.store.add_relations([relation])
        assert count == 0, "Should not store relation with missing source"

        relations = self.store.all_relations()
        assert len(relations) == 0, "No relations should be stored"

    def test_relation_missing_target_endpoint_not_stored(self):
        """Relation with missing target entity should be rejected."""
        # Add source entity only
        source = Entity(
            id="source-entity",
            name="Source Entity",
            type=EntityType.METHOD,
            paper_ids=["paper1"],
        )
        self.store.add_entities([source])

        # Try to add relation with missing target
        relation = Relation(
            id="source-entity_relates_to_missing-target_paper1",
            source_id="source-entity",
            relation_type=RelationType.INTRODUCES,
            target_id="missing-target",
            evidence="Test evidence",
            paper_id="paper1",
        )

        count = self.store.add_relations([relation])
        assert count == 0, "Should not store relation with missing target"

        relations = self.store.all_relations()
        assert len(relations) == 0, "No relations should be stored"

    def test_relation_both_endpoints_missing_not_stored(self):
        """Relation with both endpoints missing should be rejected."""
        relation = Relation(
            id="missing1_relates_to_missing2_paper1",
            source_id="missing1",
            relation_type=RelationType.INTRODUCES,
            target_id="missing2",
            evidence="Test evidence",
            paper_id="paper1",
        )

        count = self.store.add_relations([relation])
        assert count == 0, "Should not store relation with both endpoints missing"

        relations = self.store.all_relations()
        assert len(relations) == 0, "No relations should be stored"

    def test_same_entities_different_papers_both_kept(self):
        """Two papers reporting same entity pair should both be kept with different IDs."""
        # Add entities
        source = Entity(
            id="source-entity",
            name="Source Entity",
            type=EntityType.METHOD,
            paper_ids=["paper1", "paper2"],
        )
        target = Entity(
            id="target-entity",
            name="Target Entity",
            type=EntityType.TECHNIQUE,
            paper_ids=["paper1", "paper2"],
        )
        self.store.add_entities([source, target])

        # Add relation from paper1
        rel1 = Relation(
            id="source-entity_introduces_target-entity_paper1",
            source_id="source-entity",
            relation_type=RelationType.INTRODUCES,
            target_id="target-entity",
            evidence="Paper 1 introduces this technique",
            paper_id="paper1",
            chunk_ids=["chunk1"],
        )

        # Add relation from paper2 (same entities, different paper)
        rel2 = Relation(
            id="source-entity_introduces_target-entity_paper2",
            source_id="source-entity",
            relation_type=RelationType.INTRODUCES,
            target_id="target-entity",
            evidence="Paper 2 also introduces this technique",
            paper_id="paper2",
            chunk_ids=["chunk2"],
        )

        count = self.store.add_relations([rel1, rel2])
        assert count == 2, "Both relations should be stored"

        relations = self.store.all_relations()
        assert len(relations) == 2, "Both relations should be retrievable"

        # Verify both have correct provenance
        paper_ids = [r.paper_id for r in relations]
        assert "paper1" in paper_ids
        assert "paper2" in paper_ids

        # Verify relation IDs are different (include paper_id)
        rel_ids = [r.id for r in relations]
        assert len(set(rel_ids)) == 2, "Relation IDs should be unique per paper"

    def test_valid_both_endpoints_stored_and_retrievable(self):
        """Valid relation with both endpoints should be stored and retrievable."""
        # Add both entities
        source = Entity(
            id="method-a",
            name="Method A",
            type=EntityType.METHOD,
            paper_ids=["paper1"],
        )
        target = Entity(
            id="technique-b",
            name="Technique B",
            type=EntityType.TECHNIQUE,
            paper_ids=["paper1"],
        )
        self.store.add_entities([source, target])

        # Add valid relation
        relation = Relation(
            id="method-a_improves_technique-b_paper1",
            source_id="method-a",
            relation_type=RelationType.IMPROVES,
            target_id="technique-b",
            evidence="Method A improves Technique B",
            paper_id="paper1",
            chunk_ids=["chunk1"],
            weight=1.5,
        )

        count = self.store.add_relations([relation])
        assert count == 1, "Valid relation should be stored"

        relations = self.store.all_relations()
        assert len(relations) == 1, "Should be retrievable"

        retrieved = relations[0]
        assert retrieved.source_id == "method-a"
        assert retrieved.target_id == "technique-b"
        assert retrieved.relation_type == RelationType.IMPROVES
        assert retrieved.paper_id == "paper1"
        assert retrieved.evidence == "Method A improves Technique B"
        assert retrieved.weight == 1.5

    def test_mixed_valid_and_invalid_relations_only_valid_stored(self):
        """When adding multiple relations, only valid ones should be stored."""
        # Add entities
        source = Entity(
            id="entity-a",
            name="Entity A",
            type=EntityType.METHOD,
            paper_ids=["paper1"],
        )
        target = Entity(
            id="entity-b",
            name="Entity B",
            type=EntityType.TECHNIQUE,
            paper_ids=["paper1"],
        )
        self.store.add_entities([source, target])

        # Mix of valid and invalid relations
        valid_rel = Relation(
            id="entity-a_introduces_entity-b_paper1",
            source_id="entity-a",
            relation_type=RelationType.INTRODUCES,
            target_id="entity-b",
            evidence="Valid relation",
            paper_id="paper1",
        )
        invalid_rel = Relation(
            id="entity-a_introduces_missing_paper1",
            source_id="entity-a",
            relation_type=RelationType.INTRODUCES,
            target_id="missing-entity",
            evidence="Invalid - missing target",
            paper_id="paper1",
        )

        count = self.store.add_relations([valid_rel, invalid_rel])
        assert count == 1, "Only valid relation should be stored"

        relations = self.store.all_relations()
        assert len(relations) == 1
        assert relations[0].id == "entity-a_introduces_entity-b_paper1"

    def test_edge_order_independence(self):
        """Adding entity after relation attempt should not retroactively add relation."""
        # Try to add relation before entities exist
        rel = Relation(
            id="a_relates_b_paper1",
            source_id="a",
            relation_type=RelationType.RELATED_TO,
            target_id="b",
            evidence="Test",
            paper_id="paper1",
        )
        count1 = self.store.add_relations([rel])
        assert count1 == 0

        # Now add entities
        self.store.add_entities([
            Entity(id="a", name="A", type=EntityType.METHOD, paper_ids=["paper1"]),
            Entity(id="b", name="B", type=EntityType.METHOD, paper_ids=["paper1"]),
        ])

        # Relation should still not exist (was not stored)
        relations = self.store.all_relations()
        assert len(relations) == 0

        # Need to explicitly add relation again
        count2 = self.store.add_relations([rel])
        assert count2 == 1

        relations = self.store.all_relations()
        assert len(relations) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])