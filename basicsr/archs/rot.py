import torch
from scipy.linalg import hadamard

# 定义四元数参数类
class QuaternionParameter(torch.nn.Module):
    def __init__(self, num_quaternions):
        """
        初始化四元数参数
        :param num_quaternions: 四元数的数量
        """
        super(QuaternionParameter, self).__init__()
        self.num_quaternions = num_quaternions

        # 初始化为随机Hadamard矩阵元素并归一化
        hadamard_size = 2 ** (num_quaternions - 1).bit_length()
        h_matrix = hadamard(hadamard_size)[:num_quaternions, :4]  # 提取前 num_quaternions 行和前 4 列
        h_tensor = torch.tensor(h_matrix, dtype=torch.float32)

        # 归一化为单位四元数
        self.quaternions = torch.nn.Parameter(
            h_tensor / h_tensor.norm(dim=1, keepdim=True)
        )

    def forward(self):
        # 四元数在 forward 时自动归一化以确保约束
        return self.quaternions / self.quaternions.norm(dim=1, keepdim=True)

# 示例：创建 3 个可学习的四元数
num_quaternions = 3
model = QuaternionParameter(num_quaternions)