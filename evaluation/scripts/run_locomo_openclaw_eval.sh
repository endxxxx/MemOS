#!/bin/bash

# Common parameters for openclaw evaluation
VERSION="default"
LIB="openclaw"


echo "Running locomo_openclaw_eval.py..."
python scripts/locomo/locomo_openclaw_eval.py --client_type $LIB --version $VERSION
if [ $? -ne 0 ]; then
    echo "Error running locomo_openclaw_eval.py"
    exit 1
fi

echo "Running locomo_openclaw_metrics.py..."
python scripts/locomo/locomo_openclaw_metrics.py --client_type $LIB --version $VERSION
if [ $? -ne 0 ]; then
    echo "Error running locomo_openclaw_metrics.py"
    exit 1
fi

echo "All scripts completed successfully!"
