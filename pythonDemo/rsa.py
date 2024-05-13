# 实现一个rsa算法，对字符串“HelloWorld”进行加密和解密。

import random

# 简单的欧几里得算法求最大公约数
def gcd(a, b):
    while b != 0:   
        a, b = b, a % b
    return a

# 求模逆运算
def mod_inverse(e, phi):
    for x in range(1, phi):
        if (e * x) % phi == 1:
            return x
    raise Exception('Modular inverse not found')


# 欧拉函数
def euler_totient(n):
    phi = n
    p_factor, exponent = 2, 0
    while p_factor * p_factor <= n:
        while n % p_factor == 0:
            n //= p_factor
            exponent += 1
        if p_factor == 2:
            p_factor += 1
        else:
            p_factor += 2
    if n > 1:
        exponent += 1
    return n * (p_factor - 1) * (pow(p_factor, exponent) - 1) // p_factor

# 随机生成两个不同的大质数
def generate_primes():
    primes = []
    while len(primes) < 2:
        p = random.randrange(100, 300)
        for div in range(2, int(p ** 0.5) + 1):
            if p % div == 0:
                break
        else:
            primes.append(p)

    primes[0], primes[1] = sorted(primes)
    return primes


# RSA密钥生成
def generate_keys():
    p, q = generate_primes()
    n = p * q
    phi = euler_totient(n)
    e = random.randrange(1, phi)
    g = gcd(e, phi)
    while g != 1:
        e += 1
        g = gcd(e, phi)
    d = mod_inverse(e, phi)
    return ((e, n), (d, n))

# RSA加密
def rsa_encrypt(message, public_key):
    e, n = public_key
    encrypted_msg = [pow(ord(char), e, n) for char in message]
    return encrypted_msg

# RSA解密
def rsa_decrypt(encrypted_msg, private_key):
    d, n = private_key
    message = ''.join([chr(pow(char, d, n)) for char in encrypted_msg])
    return message

# 主函数
def main():
    message = "HelloWorld"
    public_key, private_key = generate_keys()
    print(f'Public Key: {public_key}')
    print(f'Private Key: {private_key}')
    
    encrypted_msg = rsa_encrypt(message, public_key)
    print(f'Encrypted Message: {encrypted_msg}')
    
    decrypted_msg = rsa_decrypt(encrypted_msg, private_key)
    print(f'Decrypted Message: {decrypted_msg}')

if __name__ == "__main__":
    main()  