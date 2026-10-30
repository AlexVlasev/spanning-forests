import argparse

import pytest

from graph import complete_graph
from helpers import adjacency
from main import main, parse_graph, parse_vertices


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
    assert main(['111', '123', '--graph', 'K5', '--reduce', '--verify', '--quiet']) == 0
    out = capsys.readouterr().out
    assert 'Dimension: 3' in out
    assert 'Verified symbolically' in out
    assert 'phi_111 * phi_123 =' in out
    assert '* phi_122 * phi_121' in out


def test_main_with_custom_graph_and_marked_vertices(capsys):
    args = ['111', '123', '--graph', '0-1,1-2,2-3,3-0,0-2', '--marked', '0,1,3', '--reduce', '-q']
    assert main(args) == 0
    assert 'Dimension:' in capsys.readouterr().out


@pytest.mark.parametrize(
    'args',
    [
        ['111', '1234'],  # P and Q of different lengths
        ['111', '123', '--marked', '0,1'],  # wrong number of marked vertices
        ['111', '123', '--graph', 'K3', '--marked', '0,1,5'],  # vertex not in the graph
    ],
)
def test_main_rejects_inconsistent_arguments(args):
    with pytest.raises(SystemExit):
        main(args)
