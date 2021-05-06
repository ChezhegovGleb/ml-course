import numpy as np
from copy import deepcopy

class Relu:
    def __init__(self, alpha):
        self.alpha = alpha
        self.result_up = []
        self.tensor_in = []

    def up(self, tensor):
        self.tensor_in = deepcopy(tensor)
        d = tensor.shape[0]
        h = tensor.shape[1]
        w = tensor.shape[2]
        self.result_up = deepcopy(tensor)
        for k in range(d):
            for i in range(h):
                for j in range(w):
                    if self.result_up[k][i][j] < 0:
                        self.result_up[k][i][j] /= alpha
        return deepcopy(self.result_up)

    def down(self, tensor):
        d = self.tensor_in.shape[0]
        h = self.tensor_in.shape[1]
        w = self.tensor_in.shape[2]
        result = np.zeros((self.tensor_in.shape[0], self.tensor_in.shape[1], self.tensor_in.shape[2]))
        for k in range(d):
            for i in range(h):
                for j in range(w):
                    try:
                        if self.result_up[k][i][j] < 0:
                            result[k][i][j] = tensor[k][i][j] / alpha
                        else:
                            result[k][i][j] = tensor[k][i][j]
                    except Exception:
                        pass
        return result

    def printDer(self):
        pass

class Pool:
    def __init__(self, sz):
        self.sz = sz
        self.result_up = []
        self.tensor_in = []

    def up(self, tensor):
        self.tensor_in = deepcopy(tensor)
        self.in_d = tensor.shape[0]
        self.in_h = tensor.shape[1]
        self.in_w = tensor.shape[2]
        d = tensor.shape[0]
        h = (tensor.shape[1] - sz) // sz + 1
        w = (tensor.shape[2] - sz) // sz + 1
        self.result_up = np.array([float("-inf")] * max(1, d * h * w)).reshape((max(1, d), max(1, h), max(1, w)))
        for k in range(d):
            for i in range(h):
                for j in range(w):
                    for in_i in range(sz):
                        for in_j in range(sz):
                            try:
                                self.result_up[k][i][j] = max(self.result_up[k][i][j], tensor[k][i * sz + in_i][j * sz + in_j])
                            except:
                                pass
        return deepcopy(self.result_up)

    def down(self, tensor):
        # print("Pool")
        d = self.in_d
        h = self.in_h
        w = self.in_w
        result = np.zeros((d, h, w))
        for k in range(tensor.shape[0]):
            for i in range(tensor.shape[1]):
                for j in range(tensor.shape[2]):
                    max_val = float("-inf")
                    for in_i in range(self.sz):
                        for in_j in range(self.sz):
                            try:
                                if i * self.sz + in_i < self.in_h and j * self.sz + in_j < self.in_w:
                                    max_val = max(max_val, self.tensor_in[k][i * self.sz + in_i][j * self.sz + in_j])
                            except Exception:
                                pass
                    for in_i in range(self.sz):
                        for in_j in range(self.sz):
                            try:
                                if i * self.sz + in_i < self.in_h and j * self.sz + in_j < self.in_w:
                                    if self.tensor_in[k][i * self.sz + in_i][j * self.sz + in_j] == max_val:
                                        result[k][i * self.sz + in_i][j * self.sz + in_j] += tensor[k][i][j]
                            except Exception:
                                pass
        return result

    def printDer(self):
        pass

class Bias:
    def __init__(self, bs):
        self.bs = deepcopy(bs)
        self.tensor_in = []
        self.result_up = []
        self.outputs = []

    def up(self, tensor):
        self.tensor_in = deepcopy(tensor)
        d = tensor.shape[0]
        h = tensor.shape[1]
        w = tensor.shape[2]
        self.result_up = deepcopy(tensor)
        for k in range(d):
            for i in range(h):
                for j in range(w):
                    self.result_up[k][i][j] += self.bs[k]
        return deepcopy(self.result_up)

    def down(self, tensor):
        # print("Bias")
        d = self.tensor_in.shape[0]
        h = self.tensor_in.shape[1]
        w = self.tensor_in.shape[2]
        for k in range(d):
            sum = 0
            for i in range(h):
                for j in range(w):
                    try:
                        sum += tensor[k][i][j]
                    except Exception:
                        pass
            self.outputs.append(sum)
        return deepcopy(tensor)

    def printDer(self):
        print(" ".join(map(str, self.outputs)))

class Cnvm:
    def __init__(self, h, k, s, p, kernel):
        self.h = h
        self.k = k
        self.s = s
        self.p = p
        self.kernel = kernel
        self.result = []
        self.calc_kernel = []

    def up(self, tensor):
        self.tensor_in = deepcopy(tensor)
        d = tensor.shape[0]
        h = tensor.shape[1]
        w = tensor.shape[2]
        out_d = tensor.shape[0]
        out_h = (tensor.shape[1] + 2 * self.p - self.k) // self.s + 1
        out_w = (tensor.shape[2] + 2 * self.p - self.k) // self.s + 1
        self.result_up = np.zeros((out_d, out_h, out_w))
        for k in range(out_d):
            for i in range(out_h):
                for j in range(out_w):
                    for in_k in range(d):
                        for in_i in range(self.k):
                            for in_j in range(self.k):
                                ind_h = abs(-self.p + i * self.s + in_i)
                                ind_w = abs(-self.p + j * self.s + in_j)
                                while ind_h >= tensor.shape[1]:
                                    ind_h = abs(h - (ind_h % h) - 2)
                                while ind_w >= tensor.shape[2]:
                                    ind_w = abs(w - (ind_w % w) - 2)
                                try:
                                    self.result_up[k][i][j] += self.kernel[k][in_k][in_i][in_j] * tensor[in_k][ind_h][ind_w]
                                except Exception:
                                    pass
        return deepcopy(self.result_up)

    def down(self, tensor):
        # print("Cnvm")
        d = tensor.shape[0]
        h = tensor.shape[1]
        w = tensor.shape[2]
        out_d = self.h
        out_h = (self.tensor_in.shape[1] + 2 * self.p - self.k) // self.s + 1
        out_w = (self.tensor_in.shape[2] + 2 * self.p - self.k) // self.s + 1
        self.calc_kernel = np.zeros((self.h, self.tensor_in.shape[0], self.k, self.k))
        result = np.zeros((max(1, self.tensor_in.shape[0]), max(1, self.tensor_in.shape[1]), max(1, self.tensor_in.shape[2])))
        for k in range(out_d):
            for i in range(out_h):
                for j in range(out_w):
                    for in_k in range(self.tensor_in.shape[0]):
                        for in_i in range(self.k):
                            for in_j in range(self.k):
                                ind_h = abs(-self.p + i * self.s + in_i)
                                ind_w = abs(-self.p + j * self.s + in_j)
                                while ind_h >= self.tensor_in.shape[1]:
                                    ind_h = abs(self.tensor_in.shape[1] - (ind_h % h) - 2)
                                while ind_w >= self.tensor_in.shape[2]:
                                    ind_w = abs(self.tensor_in.shape[2] - (ind_w % w) - 2)
                                try:
                                    self.calc_kernel[k][in_k][in_i][in_j] += tensor[k][i][j] * self.tensor_in[in_k][ind_h][ind_w]
                                    result[in_k][ind_h][ind_w] += tensor[k][i][j] * self.kernel[k][in_k][in_i][in_j]
                                except Exception:
                                    pass

        return result

    def printDer(self):
        for i in range(self.calc_kernel.shape[0]):
            for j in range(self.calc_kernel.shape[1]):
                for k in range(self.calc_kernel.shape[2]):
                    for g in range(self.calc_kernel.shape[3]):
                        print(self.calc_kernel[i][j][k][g], end=" ")
        print()

class Cnve:
    def __init__(self, h, k, s, p, kernel):
        self.h = h
        self.k = k
        self.s = s
        self.p = p
        self.kernel = deepcopy(kernel)
        self.result = []
        self.calc_kernel = []

    def up(self, tensor):
        self.tensor_in = deepcopy(tensor)
        d = tensor.shape[0]
        h = tensor.shape[1]
        w = tensor.shape[2]
        out_d = tensor.shape[0]
        out_h = (tensor.shape[1] + 2 * self.p - self.k) // self.s + 1
        out_w = (tensor.shape[2] + 2 * self.p - self.k) // self.s + 1
        self.result_up = np.zeros((out_d, out_h, out_w))
        for k in range(out_d):
            for i in range(out_h):
                for j in range(out_w):
                    for in_k in range(d):
                        for in_i in range(self.k):
                            for in_j in range(self.k):
                                ind_h = -self.p + i * self.s + in_i
                                ind_w = -self.p + j * self.s + in_j
                                ind_h = max(0, min(ind_h, tensor.shape[1] - 1))
                                ind_w = max(0, min(ind_w, tensor.shape[2] - 1))
                                try:
                                    self.result_up[k][i][j] += self.kernel[k][in_k][in_i][in_j] * tensor[in_k][ind_h][ind_w]
                                except Exception:
                                    pass
        return self.result_up

    def down(self, tensor):
        out_d = self.h
        out_h = (self.tensor_in.shape[1] + 2 * self.p - self.k) // self.s + 1
        out_w = (self.tensor_in.shape[2] + 2 * self.p - self.k) // self.s + 1
        self.calc_kernel = np.zeros((self.h, self.tensor_in.shape[0], self.k, self.k))
        result = np.zeros((max(1, self.tensor_in.shape[0]), max(1, self.tensor_in.shape[1]), max(1, self.tensor_in.shape[2])))
        for k in range(out_d):
            for i in range(out_h):
                for j in range(out_w):
                    for in_k in range(self.tensor_in.shape[0]):
                        for in_i in range(self.k):
                            for in_j in range(self.k):
                                ind_h = -self.p + i * self.s + in_i
                                ind_w = -self.p + j * self.s + in_j
                                try:
                                    ind_h = max(0, min(ind_h, self.tensor_in.shape[1] - 1))
                                    ind_w = max(0, min(ind_w, self.tensor_in.shape[2] - 1))
                                    self.calc_kernel[k][in_k][in_i][in_j] += tensor[k][i][j] * self.tensor_in[in_k][ind_h][ind_w]
                                    result[in_k][ind_h][ind_w] += tensor[k][i][j] * self.kernel[k][in_k][in_i][in_j]
                                except Exception:
                                    pass

        return result

    def printDer(self):
        for i in range(self.calc_kernel.shape[0]):
            for j in range(self.calc_kernel.shape[1]):
                for k in range(self.calc_kernel.shape[2]):
                    for g in range(self.calc_kernel.shape[3]):
                        print(self.calc_kernel[i][j][k][g], end=" ")
        print()

class Cnvc:
    def __init__(self, h, k, s, p, kernel):
        self.h = h
        self.k = k
        self.s = s
        self.p = p
        self.kernel = kernel
        self.result = []
        self.calc_kernel = []

    def up(self, tensor):
        self.tensor_in = deepcopy(tensor)
        d = tensor.shape[0]
        h = tensor.shape[1]
        w = tensor.shape[2]
        out_d = tensor.shape[0]
        out_h = (tensor.shape[1] + 2 * self.p - self.k) // self.s + 1
        out_w = (tensor.shape[2] + 2 * self.p - self.k) // self.s + 1
        self.result_up = np.zeros((out_d, out_h, out_w))
        for k in range(out_d):
            for i in range(out_h):
                for j in range(out_w):
                    for in_k in range(d):
                        for in_i in range(self.k):
                            for in_j in range(self.k):
                                ind_h = -self.p + i * self.s + in_i
                                ind_w = -self.p + j * self.s + in_j
                                try:
                                    ind_h = (1e1 * tensor.shape[1] + ind_h) % tensor.shape[1]
                                    ind_w = (1e1 * tensor.shape[2] + ind_w) % tensor.shape[2]
                                    self.result_up[k][i][j] += self.kernel[k][in_k][in_i][in_j] * tensor[in_k][ind_h][ind_w]
                                except Exception:
                                    pass
        return deepcopy(self.result_up)

    def down(self, tensor):
        d = tensor.shape[0]
        h = tensor.shape[1]
        w = tensor.shape[2]
        out_d = self.h
        out_h = (self.tensor_in.shape[1] + 2 * self.p - self.k) // self.s + 1
        out_w = (self.tensor_in.shape[2] + 2 * self.p - self.k) // self.s + 1
        self.calc_kernel = np.zeros((self.h, self.tensor_in.shape[0], self.k, self.k))
        result = np.zeros((max(1, self.tensor_in.shape[0]), max(1, self.tensor_in.shape[1]), max(1, self.tensor_in.shape[2])))
        for k in range(out_d):
            for i in range(out_h):
                for j in range(out_w):
                    for in_k in range(self.tensor_in.shape[0]):
                        for in_i in range(self.k):
                            for in_j in range(self.k):
                                ind_h = -self.p + i * self.s + in_i
                                ind_w = -self.p + j * self.s + in_j
                                ind_h = (1e1 * self.tensor_in.shape[1] + ind_h) % self.tensor_in.shape[1]
                                ind_w = (1e1 * self.tensor_in.shape[2] + ind_w) % self.tensor_in.shape[2]

                                try:
                                    self.calc_kernel[k][in_k][in_i][in_j] += tensor[k][i][j] * self.tensor_in[in_k][ind_h][ind_w]
                                    result[in_k][ind_h][ind_w] += tensor[k][i][j] * self.kernel[k][in_k][in_i][in_j]
                                except Exception:
                                    pass

        return result

    def printDer(self):
        for i in range(self.calc_kernel.shape[0]):
            for j in range(self.calc_kernel.shape[1]):
                for k in range(self.calc_kernel.shape[2]):
                    for g in range(self.calc_kernel.shape[3]):
                        print(self.calc_kernel[i][j][k][g], end=" ")
        print()

if __name__ == '__main__':
    matrix = list(map(float, input().split()))
    n0 = int(matrix[0])
    d0 = int(matrix[1])
    curD = d0
    tensor = np.array(matrix[2:]).reshape((d0, n0, n0))

    L = int(input())
    up_transforms = []

    for i in range(L):
        request = list(map(str, input().split()))
        type = request[0]
        request = request[1:]

        if type == "relu":
            alpha = float(request[0])
            up_transforms.append(Relu(alpha))
        elif type == "pool":
            sz = int(request[0])
            up_transforms.append(Pool(sz))
        elif type == "bias":
            for i in range(len(request)):
                request[i] = float(request[i])
            up_transforms.append(Bias(request))
        elif type == "cnvm":
            h = int(request.pop(0))
            k = int(request.pop(0))
            s = int(request.pop(0))
            p = int(request.pop(0))

            for i in range(len(request)):
                request[i] = float(request[i])
            kernel = np.array(request).reshape((h, curD, k, k))
            up_transforms.append(Cnvm(h, k, s, p, kernel))
            curD = h
        elif type == "cnve":
            h = int(request.pop(0))
            k = int(request.pop(0))
            s = int(request.pop(0))
            p = int(request.pop(0))

            for i in range(len(request)):
                request[i] = float(request[i])
            kernel = np.array(request).reshape((h, curD, k, k))
            up_transforms.append(Cnve(h, k, s, p, kernel))
            curD = h
        elif type == "cnvc":
            h = int(request.pop(0))
            k = int(request.pop(0))
            s = int(request.pop(0))
            p = int(request.pop(0))

            for i in range(len(request)):
                request[i] = float(request[i])
            kernel = np.array(request).reshape((h, curD, k, k))
            up_transforms.append(Cnvc(h, k, s, p, kernel))
            curD = h

    for vertex in up_transforms:
        tensor = vertex.up(tensor)

    answer = deepcopy(tensor)

    for i in range(answer.shape[0]):
        for j in range(answer.shape[1]):
            for k in range(answer.shape[2]):
                print(answer[i][j][k], end=" ")

    down_tensor = np.array(list(map(float, input().split())))
    down_tensor = down_tensor.reshape((max(1, tensor.shape[0]), max(1, tensor.shape[1]), max(1, tensor.shape[2])))

    for i in range(len(up_transforms) - 1, -1, -1):
        down_tensor = up_transforms[i].down(down_tensor)

    for i in range(down_tensor.shape[0]):
        for j in range(down_tensor.shape[1]):
            for k in range(down_tensor.shape[2]):
                print(down_tensor[i][j][k], end=" ")
    print()

    for arr in up_transforms:
        arr.printDer()
