import cProfile, pstats

import main


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print("Usage: python {} <matlab file> <number of samples per image>".format(sys.argv[0]))
        sys.exit(1)

    matlab_file = sys.argv[1]
    num_of_samples_per_image = int(sys.argv[2])

    prof = cProfile.Profile()
    main.run(matlab_file, num_of_samples_per_image, prof)
    stats = pstats.Stats(prof)
    stats.strip_dirs().sort_stats('time').reverse_order().print_stats()
