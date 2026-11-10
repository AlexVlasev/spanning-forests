from polynomial import MultilinearPolynomial, Polynomial, edge_indices

VARIABLES = ['x', 'y', 'z']


def test_edge_indices():
    assert edge_indices(0) == ()
    assert edge_indices(0b1) == (0,)
    assert edge_indices(0b101100) == (2, 3, 5)


X = Polynomial(['x'], {(1,): 1})


def test_comparison_with_numbers():
    zero, one = Polynomial(['x'], {}), Polynomial(['x'], {(0,): 1})
    assert zero == 0
    assert zero != 7
    assert one == 1
    assert one != 2
    assert X != 1
    assert X != 0
    assert Polynomial(['x'], {(1,): 0}) == 0


def test_cancelled_terms_do_not_affect_equality():
    assert (X + 1) * (X - 1) == X * X - 1
    assert X + 0 == X
    assert X != X + 1


def test_comparison_with_other_types():
    assert X != 'x'
    assert X is not None and X != None  # noqa: E711


def test_comparison_of_collected_coefficients():
    variables = ['x', 'y']
    poly = Polynomial(variables, {(1, 0): 2, (1, 1): 1})
    assert poly.collect(1) == poly.collect(1)
    assert poly.collect(1) != Polynomial(variables, {(1, 0): 2}).collect(1)


def test_to_polynomial():
    poly = MultilinearPolynomial(VARIABLES, {0b011: 2, 0b100: -1, 0b000: 5})
    expected = Polynomial(VARIABLES, {(1, 1, 0): 2, (0, 0, 1): -1, (0, 0, 0): 5})
    assert poly.to_polynomial() == expected


def test_len_eq_and_scalar_multiplication():
    poly = MultilinearPolynomial(VARIABLES, {0b011: 2, 0b100: -1})
    assert len(poly) == 2
    assert poly == MultilinearPolynomial(VARIABLES, {0b100: -1, 0b011: 2})
    assert poly != MultilinearPolynomial(VARIABLES, {0b011: 2})
    assert poly * 3 == 3 * poly == MultilinearPolynomial(VARIABLES, {0b011: 6, 0b100: -3})
