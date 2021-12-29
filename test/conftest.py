
import os
import sys
import glob

# prepend the local build (if any) to the path.
for local_build_path in glob.glob(os.path.join("build", "lib.*")):
    sys.path.insert(0, local_build_path)
