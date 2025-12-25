#!/bin/bash
cd /root/Project/SFA
export PYTHONPATH=/root/Project/SFA
USE_GROUNDED_SAM=1 python tests/test_segmenter.py