class Matrix:
    def __init__(self, array):
        self.array = array
        self.height = len(array)
        self.width = len(array[0])
    
    def __add__(self, other):
        array = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(self.array[i][j] + other.array[i][j])
            array.append(row)
        return Matrix(array)
    
    def remove_row(self, index):
        new_array = [row for i, row in enumerate(self.array) if i != index]
        return Matrix(new_array)
    
    def remove_col(self, index):
        new_array = [[elem for j, elem in enumerate(row) if j != index] for row in self.array]
        return Matrix(new_array)
    
    def det(self):
        if self.height != self.width:
            raise Exception('The matrix is not square')
        if self.height == 1:
            return self.array[0][0]
        
        sign = -1
        matrix = self.remove_row(0)
        total = 0
        for i, elem in enumerate(self.array[0]):
            sign *= -1
            if elem == 0:
                continue
            total += matrix.remove_col(i).det() * elem * sign
        return total
    
    def transpose(self):
        array = [[self.array[j][i] for j in range(self.height)] for i in range(self.width)]
        return Matrix(array)
    
    def __neg__(self):
        array = [[-elem for elem in row] for row in self.array]
        return Matrix(array)
    
    def __str__(self):
        return str(self.array)
    
    def __repr__(self):
        return f'Matrix({self.height}, {self.width}, {self.array})'

    @staticmethod
    def expand_block(A, B, C):
        """
        Block matrix:

            |A B|
            |C 0|
        
        D is optional (filled with zeros by default)
        A, B same height
        A, C same width
        """

        if len(A) != len(B):
            raise Exception('A and B of unequal height')
        if len(A[0]) != len(C[0]):
            raise Exception('A and C of unequal width')

        array = []
        zeros = [0 for i in range(len(B[0]))]
        for a, b in zip(A, B):
            row = [*a, *b]
            array.append(row)
        for c in C:
            row = [*c, *zeros]
            array.append(row)
        return Matrix(array)
                
                    



if __name__ == '__main__':
    matrix = Matrix([[1, 1], [-2, 1]])
    print(matrix.det())
    print(matrix.transpose())
    print(-matrix)

    A = [[1, 0], [0, 1]]
    B = [[2, 3, 4], [5, 6, 7]]
    C = Matrix(B).transpose().array

    aa = Matrix(A)
    print(aa.expand_block(A, B, C))
