import cProfile
import pstats

# Step 1: Profile and save binary stats
#cProfile.run('cutting_fidelity_test.py', 'output.prof')

# Step 2: Convert to readable text file
with open('readable_stats2.txt', 'w') as f:
    stats = pstats.Stats('output.prof', stream=f)
    stats.strip_dirs().sort_stats('cumulative').print_stats()
