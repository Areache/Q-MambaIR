import torch

def build_power_value(B=2, additive=True):
    base_a = [0.]
    base_b = [0.]
    base_c = [0.]
    if additive:
        if B == 2:
            for i in range(3):
                base_a.append(2 ** (-i - 1))
        elif B == 4:
            for i in range(3):
                base_a.append(2 ** (-2 * i - 1))
                base_b.append(2 ** (-2 * i - 2))
        elif B == 6:
            for i in range(3):
                base_a.append(2 ** (-3 * i - 1))
                base_b.append(2 ** (-3 * i - 2))
                base_c.append(2 ** (-3 * i - 3))
        elif B == 3:
            for i in range(3):
                if i < 2:
                    base_a.append(2 ** (-i - 1))
                else:
                    base_b.append(2 ** (-i - 1))
                    base_a.append(2 ** (-i - 2))
        elif B == 5:
            for i in range(3):
                if i < 2:
                    base_a.append(2 ** (-2 * i - 1))
                    base_b.append(2 ** (-2 * i - 2))
                else:
                    base_c.append(2 ** (-2 * i - 1))
                    base_a.append(2 ** (-2 * i - 2))
                    base_b.append(2 ** (-2 * i - 3))
        else:
            pass
    else:
        for i in range(2 ** B - 1):
            base_a.append(2 ** (-i - 1))
    values = []
    for a in base_a:
        for b in base_b:
            for c in base_c:
                values.append((a + b + c))
    values = torch.Tensor(list(set(values)))
    values = values.mul(1.0 / torch.max(values))
    return values
# Example usage
if __name__ == "__main__":
    print(build_power_value(B=4, additive=True))