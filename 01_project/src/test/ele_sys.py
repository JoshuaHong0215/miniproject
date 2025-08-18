import time

def countdown(seconds, message=''):
    for i in range(1, seconds +1):
        print(f'{i}sec {message}')
        time.sleep(1)

print('문이열립니다')
# 4초 대기
countdown(4)

print('대기')
# 7초 대기
countdown(7)

print('문이 닫힙니다')
# 4초 대기
countdown(4)


