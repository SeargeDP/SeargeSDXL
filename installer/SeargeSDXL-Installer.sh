#To make the script executable, save it to a file (e.g., SeargeSDXL-Installer.sh) and run chmod +x SeargeSDXL-Installer.sh.
#!/bin/bash

# Check if SeargeSDXL-Installer.py exists
if [[ ! -f "SeargeSDXL-Installer.py" ]]; then
    echo "Installer scripts not correctly installed"
    echo "Please copy the files SeargeSDXL-Installer.bat and SeargeSDXL-Installer.py into the ComfyUI_windows_portable folder"
    exit 1
fi

# Check for python_embedded in current directory
if [[ -d "python_embeded" ]]; then
    echo "Installer running correctly in ComfyUI_windows_portable"
    python_embeded/python -m pip install opencv-python
    python_embeded/python SeargeSDXL-Installer.py from_batch
    exit 0
fi

# Check for python_embedded in extension directory
if [[ -d "../../../../python_embeded" ]]; then
    echo "Installer running within SeargeSDXL extension folder, this may work but is not supported"
    ../../../../python_embeded/python -m pip install opencv-python
    ../../../../python_embeded/python SeargeSDXL-Installer.py from_batch
    exit 0
fi

echo "How did you get here?"
