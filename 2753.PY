year = int(input()) # input()함수는 기본적으로 문자열 int로 변환

def yoon(i): # i는 매개변수 -> 이름 상관없음
    if (i % 4 == 0 and i % 100 != 0) or i % 400 == 0:
        return 1
    else:
        return 0

value = yoon(year) # year는 인자 -> i(매개변수)에 들어감
print(value)