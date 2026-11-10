"""Searches for identities between spanning forest polynomials of graphs."""

from spanning_forests.graph import Graph, complete_graph
from spanning_forests.identities import IdentitySolver
from spanning_forests.polynomial import MultilinearPolynomial
from spanning_forests.relations import (
    RelationCounts,
    all_shapes,
    expand_product,
    find_relations,
    format_relation,
    product_terms,
    relation_counts,
    relation_dimension,
    verify_expansion,
)
from spanning_forests.solution import Solution

__all__ = [
    'Graph',
    'IdentitySolver',
    'MultilinearPolynomial',
    'RelationCounts',
    'Solution',
    'all_shapes',
    'complete_graph',
    'expand_product',
    'find_relations',
    'format_relation',
    'product_terms',
    'relation_counts',
    'relation_dimension',
    'verify_expansion',
]
