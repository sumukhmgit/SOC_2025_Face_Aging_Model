import math

def is_prime(n):
    # Hint: A prime number is greater than 1 and has no divisors other than 1 and itself.
    #Try looping from 2 to √n and check if any number divides n evenly.
    # Write your logic here
    if n>=1:
      for i in range(2,int(math.sqrt(n))):
        if n%i==0:
          return False
      return True
    else:
      return False