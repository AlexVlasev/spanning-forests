import argparse

import pytest

from graph import complete_graph
from helpers import adjacency
from main import main, parse_graph, parse_shape, parse_vertices


def test_parse_graph():
    assert parse_graph('K4') == complete_graph(4)
    assert parse_graph('k3') == complete_graph(3)
    assert parse_graph('0-1,1-2,2-0,2-3') == adjacency(4, [(0, 1), (1, 2), (2, 0), (2, 3)])


@pytest.mark.parametrize('spec', ['K', 'Kx', '0-1,1', '0-1-2', 'a-b'])
def test_parse_graph_rejects_invalid(spec):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_graph(spec)


def test_parse_vertices():
    assert parse_vertices('0,2,5') == [0, 2, 5]
    with pytest.raises(argparse.ArgumentTypeError):
        parse_vertices('0,x')


def test_main_prints_identity(capsys):
    assert main(['111', '123', '--graph', 'K5', '--balanced', '--verify', '--quiet']) == 0
    out = capsys.readouterr().out
    assert 'Dimension: 3' in out
    assert 'Verified symbolically' in out
    assert 'phi_111 * phi_123 =' in out
    assert '* phi_122 * phi_121' in out


def test_main_with_custom_graph_and_marked_vertices(capsys):
    args = ['111', '123', '--graph', '0-1,1-2,2-3,3-0,0-2', '--marked', '0,1,3', '--balanced', '-q']
    assert main(args) == 0
    assert 'Dimension:' in capsys.readouterr().out


def test_main_with_exact_solver(capsys):
    assert main(['111', '123', '--graph', 'K5', '--balanced', '--exact-solver', '-q']) == 0
    assert 'Dimension: 3' in capsys.readouterr().out


def test_parse_shape():
    assert parse_shape('2,3') == (2, 3)
    assert parse_shape('2,2,3') == (2, 2, 3)
    with pytest.raises(argparse.ArgumentTypeError):
        parse_shape('2,x')


def test_main_with_relations(capsys):
    assert main(['--relations', '2,3', '--marked', '0,1,2,3', '--graph', 'K5', '-q']) == 0
    out = capsys.readouterr().out
    assert 'Products : 42' in out
    assert 'Dimension: 8' in out


def test_main_with_mixed_shapes(capsys):
    assert main(['--relations', '1,3', '2,2', '--marked', '0,1,2', '-q']) == 0
    assert 'Dimension: 1' in capsys.readouterr().out


def test_main_shows_relations(capsys):
    args = ['--relations', '1,3', '2,2', '--marked', '0,1,2', '--show-relations', '-q']
    assert main(args) == 0
    out = capsys.readouterr().out
    assert 'Dimension: 1' in out
    assert 'Relation 1:' in out
    assert '+ 1 * phi_111 * phi_123' in out
    assert '- 1 * phi_112 * phi_121' in out


def test_main_splits_relations_into_inherited_and_new(capsys):
    args = ['--relations', '2,3,4', '--marked', '0,1,2,3', '--new', '-q']
    assert main(args) == 0
    out = capsys.readouterr().out
    assert 'Dimension: 8' in out
    assert 'Inherited: 8' in out
    assert 'New      : 0' in out


def test_main_row_reduces_only_once_for_new_relations(capsys, monkeypatch):
    def fail(*args):
        raise AssertionError('relation_counts already has the dimension')

    monkeypatch.setattr('main.relation_dimension', fail)
    assert main(['--relations', '2,3', '--marked', '0,1,2,3', '--new', '-q']) == 0
    assert 'Dimension: 8' in capsys.readouterr().out


def test_main_sweeps_complete_graphs(capsys):
    assert main(['--relations', '1,3', '2,2', '--marked', '0,1,2', '--sweep', '5', '-q']) == 0
    rows = [line.split() for line in capsys.readouterr().out.splitlines() if line.startswith('K')]
    assert rows == [
        ['K3', '7', '1', '0', '1'],
        ['K4', '7', '1', '0', '1'],
        ['K5', '7', '1', '0', '1'],
    ]


def test_main_expands_a_product(capsys):
    args = ['--relations', '3,3,3', '--product', '1111,1234,1234', '--marked', '0,1,2,3', '-q']
    assert main([*args, '--graph', 'K6']) == 0
    out = capsys.readouterr().out
    assert 'Dimension: 0' in out
    assert 'phi_1111 * phi_1234 * phi_1234 =' in out
    assert '+ (1) * phi_1123 * phi_1213 * phi_1231' in out
    assert 'Fixd: 1123 x 1213 x 1231 = 1' in out


def test_main_verifies_an_expansion(capsys):
    args = ['--relations', '3,3,3', '--product', '1111,1234,1234', '--marked', '0,1,2,3', '-q']
    assert main([*args, '--graph', 'K6', '--verify']) == 0
    assert 'Verified symbolically' in capsys.readouterr().out


def test_main_without_an_expansion(capsys):
    args = ['--relations', '1,3', '--product', '112,121', '--marked', '0,1,2', '-q']
    assert main(args) == 1
    assert 'No Solution' in capsys.readouterr().out


@pytest.mark.parametrize(
    'args',
    [
        [],  # neither P and Q nor relations
        ['111', '123', '--show-relations'],  # showing relations without relations
        ['111', '123', '--new'],  # counting new relations without relations
        ['111', '123', '--sweep', '5'],  # a sweep without relations
        ['--relations', '2,2', '--marked', '0,1,2', '--sweep', '2'],  # a sweep that is too small
        ['111'],  # P without Q
        ['--relations', '2,3'],  # relations without marked vertices
        ['111', '--relations', '2,3', '--marked', '0,1,2'],  # relations with P
        ['--relations', '2,2', '2,3', '--marked', '0,1,2'],  # degrees that do not match
        ['111', '123', '--product', '111,123'],  # a product without relations
        ['--relations', '2,3', '--product', '111,123', '--marked', '0,1,2'],  # degrees differ
        ['--relations', '1,3', '--product', '11,12', '--marked', '0,1,2'],  # patterns too short
        ['--relations', '2,2', '--product', '111,123', '--marked', '0,1,2', '--new'],
        ['--relations', '2,2', '--marked', '0,1,2', '--product', '111,123', '--sweep', '5'],
        ['--relations', '2,3', '--marked', '0,1,2,3', '--verify'],  # nothing to verify
        ['--relations', '2,3', '--marked', '0,1,2,3', '--exact-solver'],  # always modular
        ['--relations', '2,3', '--marked', '0,1,2,3', '--balanced'],  # shapes are given
        ['--relations', '1,3', '--product', '111,123', '--marked', '0,1,2', '--exact-solver'],
        ['--relations', '1,3', '--product', '111,123', '--marked', '0,1,2', '--balanced'],
        ['2111', '1234'],  # P not in standard form
        ['1111', '1324', '--verify'],  # Q not in standard form
        ['123', '111', '--graph', 'K4'],  # no partition into 0 parts for the Q_j
        ['111', '123', '--marked', '0,1,1'],  # a repeated marked vertex
        ['111', '123', '--marked=0,1,-1'],  # a negative marked vertex
        ['--relations', '2,3', '--marked=0,1,2,-1'],  # a negative vertex with relations
        ['--relations', '2,2', '--marked', '0,1,1', '--sweep', '5'],  # repeated in a sweep
        ['--relations', '1,3', '--product', '121,112', '--marked', '0,1,1'],
        ['--relations', '1,3', '--product', '211,123', '--marked', '0,1,2'],  # product not standard
        ['--relations', '2,4', '--product', '211,123', '--marked', '0,1,2'],  # no products either
        ['111', '1234'],  # P and Q of different lengths
        ['111', '123', '--marked', '0,1'],  # wrong number of marked vertices
        ['111', '123', '--graph', 'K3', '--marked', '0,1,5'],  # vertex not in the graph
    ],
)
def test_main_rejects_inconsistent_arguments(args):
    with pytest.raises(SystemExit):
        main(args)
