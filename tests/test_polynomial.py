from polynomial import MultilinearPolynomial, Polynomial, edge_indices

VARIABLES = ['x', 'y', 'z']


def test_edge_indices():
    assert edge_indices(0) == ()
    assert edge_indices(0b1) == (0,)
    assert edge_indices(0b101100) == (2, 3, 5)


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
