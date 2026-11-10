import sympy

from spanning_forests.solution import Solution

x, y = sympy.symbols('x_0 x_1')
SOLUTION = Solution(
    ('111', '123'),
    {('122', '121'): 1 - x, ('121', '122'): x, ('122', '122'): 0, ('112', '121'): 2},
)


def test_free_parameters_and_dimension():
    assert SOLUTION.free_parameters == [x]
    assert SOLUTION.dimension == 1
    assert Solution(('11', '12'), {('12', '11'): 1}).dimension == 0


def test_report():
    report = SOLUTION.report()
    assert 'Depd: 122 x 121 = 1 - x_0' in report
    assert 'Fixd: 112 x 121 = 2' in report
    assert 'Free: 121 x 122 = x_0' in report
    assert 'Num vars : 4' in report
    assert 'Dimension: 1' in report
    assert 'Forbidden: 1' in report


def test_identity_lists_nonzero_terms():
    assert SOLUTION.identity().splitlines() == [
        'phi_111 * phi_123 =',
        '    + (1 - x_0) * phi_122 * phi_121',
        '    + (x_0) * phi_121 * phi_122',
        '    + (2) * phi_112 * phi_121',
    ]
