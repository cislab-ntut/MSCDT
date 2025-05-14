import random
import sympy as sy
import numpy as np
PRIME=2**16



def dot_product_triples(a,b, x0, x1, y0, y1,PRIME):#, Z0=0, Z1=0, X0=[], Y0=[], X1=[], Y1=[]):
    #if n > 0 & len(X0 = 0):
    #X0 = sy.Matrix([random.randint(0, 5) for _ in range(n)])
    #X1 = sy.Matrix([random.randint(0, 5) for _ in range(n)])
    #Y0 = sy.Matrix([random.randint(0, 5) for _ in range(n)])
    #Y1 = sy.Matrix([random.randint(0, 5) for _ in range(n)])
    #X0 = np.array([random.randint(0, 10) for _ in range(n)],dtype=int)
    #X1 = np.array([random.randint(0, 10) for _ in range(n)],dtype=int)
    #Y0 = np.array([random.randint(0, 10) for _ in range(n)],dtype=int)
    #Y1 = np.array([random.randint(0, 10) for _ in range(n)],dtype=int)
    X0 = np.random.randint(0,PRIME-1,(a,b))
    X1 = np.random.randint(0,PRIME-1,(a,b))
    Y0 = np.random.randint(0,PRIME-1,(b,))
    Y1 = np.random.randint(0,PRIME-1,(b,))
    T = np.random.randint(0,PRIME-1,(a,))
    Z0 = (np.dot(X0,Y1) +T)%PRIME
    Z1 = (np.dot(X1,Y0) - T)%PRIME

    p0x = (x0 + X0)%PRIME
    p0y = (y0 + Y0)%PRIME
    p1x = (x1 + X1)%PRIME
    p1y = (y1 + Y1)%PRIME

    z0=((np.dot(x0,(y0+p1y)))- (np.dot(p1x,Y0))+Z0)%PRIME
    z1=((np.dot(x1,(y1+p0y)))- (np.dot(p0x,Y1))+Z1)%PRIME
    #z0=(np.dot(x0,(y0+p1y))%PRIME)- (np.dot(Y0,np.transpose(p1x))%PRIME)+Z0
    #z1=(np.dot(x1,(y1+p0y))%PRIME)- (np.dot(Y1,np.transpose(p0x))%PRIME)+Z1

    #p0x = (x0 + X0) %PRIME
    #p0y = (y0 + Y0) %PRIME
    #p1x = (x1 + X1) %PRIME
    #p1y = (y1 + Y1) %PRIME

    #z0=(np.dot(x0,(y0+p1y)%PRIME)%PRIME)- ((np.dot(Y0,np.transpose(p1x))+np.dot(X0,Y1))%PRIME)
    #z1=(np.dot(x1,(y1+p0y)%PRIME)%PRIME)- ((np.dot(Y1,np.transpose(p0x))+np.dot(X1,Y0))%PRIME)

    #z0 = ((x0.dot((y0 + p1y)%PRIME))%PRIME - (Y0.dot(p1x.T))%PRIME + Z0)%PRIME
    #z1 = ((x1.dot((y1 + p0y)%PRIME))%PRIME - (Y1.dot(p0x.T ))%PRIME + Z1)%PRIME
    #print("z0: ", z0)
    #print("z1: ", z1)
    return z0, z1
if __name__ == "__main__":
    for i in range(1000):
        a=10
        b=20
        x=np.random.randint(0,2,(a,b))
        x0= np.random.randint(0,PRIME-1,(a,b))
        x1 = (x-x0)%PRIME
        y=np.random.randint(0,10,(b))
        y0 = np.random.randint(0,PRIME-1,(b,))
        y1 = (y-y0)%PRIME

        
        z0, z1 = dot_product_triples(a,b, x0, x1, y0, y1,PRIME)
        #print("z: ", (z0 + z1)%PRIME)
        #print("z: ", (z0 + z1)%PRIME)
        t=((z0 + z1)%PRIME)
        r=(x.dot(y))
        #print(r)
        #print(t)
        #print("\n")
        if (t.all()!=r.all()):
            print("false")
        #print("direct inner product: ", (x0 + x1).dot(y0 + y1))