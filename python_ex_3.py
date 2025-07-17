try:
    n = int(input('숫자를 입력하세요.'))
except ValueError:
    print('잘못 입력하셨습니다.')
    exit()
def num(n):
    if n <= 1 :
        return False
    for i in range(2,n):
        if n % i == 0 :
            return False
        return True
if num(n):
    print(f'True.{n}은 소수입니다.')
else:
    print('소수가 아닙니다.')