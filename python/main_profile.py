import cProfile, pstats

import main


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 4:
        print("Usage: python {} <matlab file> <number of samples per image> <forest file> [profiling stats file]".format(sys.argv[0]))
        sys.exit(1)

    matlab_file = sys.argv[1]
    num_of_samples_per_image = int(sys.argv[2])
    forest_file = sys.argv[3]

    prof = cProfile.Profile()
    main.run(matlab_file, num_of_samples_per_image, forest_file, prof)
    stats = pstats.Stats(prof)
    stats.strip_dirs().sort_stats('time').reverse_order().print_stats()
    if len(sys.argv) > 4:
        dump_file = sys.argv[4]
        stats.dump_stats(dump_file)
