import r0123456


def main():
    solver = r0123456.r0123456()

    # Available benchmark options:
    # "tour50.csv"
    # "tour250.csv"
    # "tour500.csv"
    # "tour750.csv"
    # "tour1000.csv"
    solver.optimize("benchmark/tour250.csv")


if __name__ == "__main__":
    main()
