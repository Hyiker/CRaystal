import os

def main():
    file_path = os.path.join(os.path.dirname(__file__), 'CIE_xyz_1931_2deg.csv')
    N = 95
    with open(file_path, 'r') as f:
        lines = f.readlines()

    data = [list(map(float, line.strip().split(','))) for line in lines[1:]]
    wavelengths = [row[0] for row in data]
    X = [row[1] for row in data]
    Y = [row[2] for row in data]
    Z = [row[3] for row in data]

    step = (wavelengths[-1] - wavelengths[0]) / (N - 1)
    new_wavelengths = [wavelengths[0] + i * step for i in range(N)]

    def interpolate(w, values):
        result = []
        for nw in new_wavelengths:
            for i in range(len(w) - 1):
                if w[i] <= nw <= w[i + 1]:
                    t = (nw - w[i]) / (w[i + 1] - w[i])
                    result.append(values[i] * (1 - t) + values[i + 1] * t)
                    break
        return result

    sampled_X = interpolate(wavelengths, X)
    sampled_Y = interpolate(wavelengths, Y)
    sampled_Z = interpolate(wavelengths, Z)

    cpp_result = "float kXYZTable[3][{}] = {{\n".format(N)
    cpp_result += "    {" + ", ".join(f"{x:.6f}f" for x in sampled_X) + "},\n"
    cpp_result += "    {" + ", ".join(f"{y:.6f}f" for y in sampled_Y) + "},\n"
    cpp_result += "    {" + ", ".join(f"{z:.6f}f" for z in sampled_Z) + "}\n"
    cpp_result += "};"
    print(cpp_result)

    print("Sum of Y = ", sum(Y))

if __name__ == "__main__":
    main()
