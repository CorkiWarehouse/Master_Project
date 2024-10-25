# import math
#
# while True:
#     # 获取输入并拆分为列表
#     input_values = input().split()
#
#     # 检查输入是否为空，如果为空，则退出循环
#     if not input_values:
#         break
#
#     # 将输入的第一个值转为浮点数，第二个值转为整数
#     try:
#         num = float(input_values[0])
#         iterations = int(input_values[1])
#     except (ValueError, IndexError):
#         print("输入格式不正确，请重新输入")
#         continue
#
#     # 如果第二个值小于等于0，输出0并继续下次循环
#     if iterations <= 0:
#         print(round(0, 2))
#         continue
#
#     total_sum = 0.0
#     # 进行迭代
#     for i in range(iterations):
#         total_sum += num
#         num = math.sqrt(num)
#
#     # 输出保留两位小数的结果
#     print(round(total_sum, 2))
#


# while True:
#     # 获取输入并拆分为列表
#     input_values = input().split()
#
#     # 检查输入是否为空，如果为空，则退出循环
#     if not input_values:
#         break
#
#     # 将输入的第一个值转为浮点数，第二个值转为整数
#     try:
#         num = int(input_values[0])
#         num2 = int(input_values[1])
#     except (ValueError, IndexError):
#         print("输入格式不正确，请重新输入")
#         continue
#
#     # 如果第二个值小于等于0，输出0并继续下次循环
#     if num2 <= 0:
#         print(round(0, 2))
#         continue
#     result = []
#     for i in range(num,num2+1):
#         a = i
#         sum = 0
#         while( a  // 10 != 0):
#             sum = sum + (a%10) **3
#             a = (a // 10)
#         sum = sum + (a) ** 3
#         if sum == i :
#             result.append(i)
#     if result:
#         for i in result:
#             print(i,end=" ")
#             print()
#     else:
#         print("no")




board_not_used = [[0 for _ in range(3)] for _ in range(3)]

def solution(board):
    total = 0
    used = [False] * 10
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    for i in range(len(board)):
        for j in board[i]:
            if j !=0:
                used[j] = True

    def valid(x,y,num):
        nonlocal directions
        for dx,dy in directions:
            nx,ny = x+dx,y+dy
            if 0<=nx<3 and 0<=ny<3:
                val = board[nx][ny]
                if val != 0 and abs(val - num) == 1:
                    return False
            return True

    def dfs(pos):
        nonlocal total,board
        # we have from 0 to 9 position
        if pos == 9:
            for i in range(3):
                for j in range(3):
                    if board[i][j] == 0:
                        return
                    else:
                        total+=1
            return
        else:
            x = pos // 3
            y = pos % 3
            if board[x][y] != 0:
                dfs(pos + 1)
            else:

                for num in range(1, 10):
                    if not used[num]:
                        if valid(x, y, num):
                            board[x][y] = num
                            used[num] = True
                            dfs(pos + 1)
                            board[x][y] = 0
                            used[num] = False

    dfs(0)
    return total

groups_nums = int(input())
for i in range(groups_nums):

    board = []
    for _ in range(3):
        row = list(map(int, input().split()))
        board.append(row)

    print(solution(board))


'''
2
1 8 5
4 6 3
0 2 0
1 3 5
4 6 8
2 7 0
'''












'''
100 120
300 380

'''