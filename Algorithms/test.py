import numpy as np

# 定义 velocity_option 和 state_option
velocity_option = np.round(np.arange(1 / 8, 1 + 1 / 8, 1 / 8), 3)
state_option = np.round(np.arange(0, 1, 1 / 8), 3)

print(len(velocity_option))
print(len(state_option))

# 创建一个字典来存储结果
result_mapping = {}

# 遍历所有 velocity 和 state 的组合
for state in state_option:
    for velocity in velocity_option:
        # 计算 velocity 和 state 的和
        combined_value = np.round(velocity*1/8 + state, 3)

        # 找到 state_option 中最接近的值
        closest_value = min(state_option, key=lambda x: abs(x - combined_value))

        # 将结果存储到字典中
        if combined_value not in result_mapping:
            result_mapping[combined_value] = closest_value

# 打印结果
for combined_value, closest_value in result_mapping.items():
    print(f"Combined Value: {combined_value} -> Closest State Option: {closest_value}")
