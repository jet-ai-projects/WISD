"""Score the frozen original or weak model on cached original-model trajectories."""


def main():
    from data_curation.precompute_direct_opd_scores import main as score
    score()


if __name__ == "__main__":
    main()
