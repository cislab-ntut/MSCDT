import os
import random
from Crypto.Util.number import getPrime

SECUREPARAM = 20 
PRIME = getPrime(SECUREPARAM+1, os.urandom)
PRIME_SIZE = (PRIME.bit_length()//8)+1 #PRIME bytes length ##[Socket Communication modified]

def pseudo_random_generator(seed):
    random.seed(seed)
    return random.getrandbits(SECUREPARAM) % PRIME


# def rand_num():
#     return random.randint(1, PRIME)


# def prime():
#     return PRIME


# def param():
#     return SECUREPARAM
