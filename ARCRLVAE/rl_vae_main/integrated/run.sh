
#!/bin/bash

# A simple script to run the main orchestration file.
# It assumes a virtual environment is set up and activated.

# Exit immediately if a command exits with a non-zero status.
set -e

# Run the main orchestrator script
echo "Starting the ARC Solver pipeline..."
python main_orchestrator.py

echo "Pipeline finished."
