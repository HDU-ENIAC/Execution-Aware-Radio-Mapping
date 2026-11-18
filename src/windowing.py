def tile_windows(length, core=64, halo=16):
    size = core + 2 * halo
    for lo in range(0, length, core):
        hi = min(length, lo + core)
        start = min(max(lo - halo, 0), length - size)
        yield (lo, hi, start, lo - start, hi - start)
