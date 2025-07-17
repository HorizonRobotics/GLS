#!/bin/bash
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${GREEN}Starting installation process...${NC}"
git config --global http.postBuffer 524288000

echo -e "${GREEN}Installing dependencies from requirements.txt...${NC}"
pip install -r requirements.txt --use-deprecated=legacy-resolver --default-timeout=60 || {
    echo -e "${RED}Failed to install requirements${NC}"
    exit 1
}

echo -e "${GREEN}Installing pytorch3d from GitHub...${NC}"
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" || {
    echo -e "${RED}Failed to install pytorch3d${NC}"
    exit 1
}

echo -e "${GREEN}Installing diff-plane-rasterization from GitHub...${NC}"
pip install submodules/diff-plane-rasterization || {
    echo -e "${RED}Failed to install diff-plane-rasterization${NC}"
    exit 1
}

echo -e "${GREEN}Installing nerfview from GitHub...${NC}"
pip install submodules/nerfview || {
    echo -e "${RED}Failed to install nerfview${NC}"
    exit 1
}

echo -e "${GREEN}Installing segment-anything-langsplat from GitHub...${NC}"
pip install submodules/segment-anything-langsplat || {
    echo -e "${RED}Failed to install segment-anything-langsplat${NC}"
    exit 1
}

echo -e "${GREEN}Installing simple-knn from GitHub...${NC}"
pip install submodule/simple-knn || {
    echo -e "${RED}Failed to install simple-knn${NC}"
    exit 1
}

echo -e "${GREEN}Installation completed successfully!${NC}"

