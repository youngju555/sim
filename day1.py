# 양의 정수 n이 주어졌을 때,
# 1부터 n (포함)까지의 모든 정수의 합을 계산하는 Python 함수 `sum_up_to_n(n)`을 작성하세요.
# 예를 들어, n이 5이면, 함수는 1 + 2 + 3 + 4 + 5 = 15를 반환해야 합니다.


def sum_up_to_n():
    sum = 0
    for n in range(1,n+1):
        if n >= 0 :
            sum += n
        else :
            print('양의 정수를 입력하세요')

sum_up_to_n(8)

