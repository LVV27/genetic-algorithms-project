import r0843621


def main():
    solver = r0843621.r0843621()

    # Available benchmark options:
    # "tour50.csv"
    # "tour250.csv"
    # "tour500.csv"
    # "tour750.csv"
    # "tour1000.csv"
    # solver.optimize("src/benchmark/tour50.csv")
    # solver.optimize("src/benchmark/tour250.csv")
    # solver.optimize("src/benchmark/tour500.csv")
    solver.optimize("src/benchmark/tour750.csv")
    # solver.optimize("src/benchmark/tour1000.csv")


if __name__ == "__main__":
    main()
