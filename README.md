# graph-engine

# branches
- dev : Latest development should be merged to dev branch. This is the default branch.
- phx-dev : Phoenix project release branch

If you create branches for pull requests, and they become stale, please delete them.

# documentation
[see doxygen](http://xcosdaem430/index.html)

# continuous integration
When adding code to this repo, please create your own fork/branch and submit a merge request.  
  
Merge requests will trigger a pipeline, that builds, and runs sanity tests.

# build
```
scl enable devtoolset-9 bash
source /opt/xilinx/xrt/setup.sh
mkdir build
cd build
cmake ..
make -j
```

# test
```
cd build/tests
ctest # Optionally use --verbose to see all test outputs
```

# novel ideas
- Bring your own Kernel Runner
  - Just need to define your own MyKernelRunner.hpp
  - Can support pre/postprocess kernels
- Kernel Runners will not manage internal buffers
  - No need for get_inputs() at the subgraph level (Only needed at graph level)
- Single Thread Pool can execute all Kernel Runners `execute_async` function
- Superclasses of vart::TensorBuffer allow direct sharing of xrt::bo objects
- GraphEngine provides a GraphRunner that can 
  - Globally allocate buffers based on the sequence of runners
  - Implement client facing VART APIs for THE WHOLE GRAPH not a subgraph

# use cases
- Whole App Acceleration
- Inference Server
