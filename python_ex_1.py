fun = input('기능을 입력해주세요(+,-,*,/)')
x = float(input('숫자를 입력해주세요.'))
y = float(input('숫자를 입력해주세요.'))

if fun == '+':
    print(f'{x}+{y}={x+y}')
elif fun == '-':
    print(f'{x}-{y}={x-y}')
elif fun == '*':
    print(f'{x}*{y}={x*y}')
elif fun == '/':
    try:
        print(f'{x}/{y}={x/y}')
    except ZeroDivisionError:
        print('0으로 나눌 수 없습니다.')
else:
    print('올바른 연산자를 입력하지 않았습니다.')
