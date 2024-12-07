#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${GREEN}Cleaning build directory...${NC}"

# Check if build directory exists
if [ -d "build" ]; then
    # If we're in the build directory, go up one level
    if [ "$(basename $(pwd))" = "build" ]; then
        cd ..
    fi
    
    # Remove build directory
    rm -rf build
    echo -e "${GREEN}✓ Removed build directory${NC}"
    
    # Recreate empty build directory
    mkdir build
    echo -e "${GREEN}✓ Created fresh build directory${NC}"
else
    echo -e "${RED}No build directory found${NC}"
    echo -e "${GREEN}Creating build directory...${NC}"
    mkdir build
    echo -e "${GREEN}✓ Created build directory${NC}"
fi

echo -e "${GREEN}Done!${NC}"
