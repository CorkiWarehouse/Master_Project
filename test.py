def change_data(data_str):
    year,month,day = data_str.split('-')
    return f'{day}/{month}/{year}'


str_input = input()

result = change_data(str_input)

print(result)

'''
Input:
2024-10-10

Output:
10/10/2024

'''