import random
import sympy as sy

def dot_product_triples(n, x0, x1, y0, y1):#, Z0=0, Z1=0, X0=[], Y0=[], X1=[], Y1=[]):
    #if n > 0 & len(X0 = 0):
    X0 = sy.Matrix([random.randint(0, 5) for _ in range(n)])
    X1 = sy.Matrix([random.randint(0, 5) for _ in range(n)])
    Y0 = sy.Matrix([random.randint(0, 5) for _ in range(n)])
    Y1 = sy.Matrix([random.randint(0, 5) for _ in range(n)])
    T = random.randint(0, 5)

    Z0 = X0.dot(Y1) + T
    Z1 = X1.dot(Y0) - T
    print("Z", Z0, Z1)
    print("X: ", X0, X1)
    print("Y: ", Y0, Y1)
    p0x = x0 + X0
    p0y = y0 + Y0
    p1x = x1 + X1
    p1y = y1 + Y1
    print("x0 + X0: ", p0x)
    print("y0 + Y0: ", p0y)
    print("x1 + X1: ", p1x)
    print("y1 + Y1: ", p1y)

    z0 = x0.dot(y0 + p1y) - Y0.dot(p1x) + Z0
    z1 = x1.dot(y1 + p0y) - Y1.dot(p0x) + Z1
    print("z0: ", z0)
    print("z1: ", z1)
    return z0, z1

if __name__ == "__main__":
    print("asdasd")
    x0 = sy.Matrix([1,1,1,1])
    x1 = sy.Matrix([0,1,2,3])
    y0 = sy.Matrix([1,1,1,1])
    y1 = sy.Matrix([-1,-1,-1,0])
    z0, z1 = dot_product_triples(4, x0, x1, y0, y1)
    print("z: ", z0 + z1)

    print("direct inner product: ", (x0 + x1).dot(y0 + y1))