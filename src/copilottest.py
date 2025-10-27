import copilot


def main():
    solver = copilot.rcopilot()
    solver.optimize("benchmark/tour50.csv")

if __name__ == "__main__":
    main()
