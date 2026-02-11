#!/bin/bash
# ============================================================================
# 凡人修仙传 Trailer V2 — Cloud Deployment (Vast.ai)
#
# Automated workflow:
#   1. Search for RTX 5090 / 4090 instance
#   2. Create instance with PyTorch + diffusers
#   3. Upload code + narration audio
#   4. Run portrait generation + I2V shot generation
#   5. Download results
#   6. Destroy instance
#
# Prerequisites:
#   vastai set api-key YOUR_KEY
#
# Usage:
#   bash scripts/deploy_cloud.sh
# ============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/output/fanren-trailer-v2"
REMOTE_WORK="/workspace/AnimateDiff"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN} 凡人修仙传 V2 — Cloud GPU Deployment${NC}"
echo -e "${GREEN}============================================${NC}"

# ---- Step 1: Check API key ----
if ! .venv/bin/vastai show user 2>/dev/null | grep -q "email"; then
    echo -e "${RED}Error: Vast.ai API key not set.${NC}"
    echo "Run: .venv/bin/vastai set api-key YOUR_KEY"
    exit 1
fi
echo -e "${GREEN}✓ Vast.ai authenticated${NC}"

# ---- Step 2: Find a GPU instance ----
echo -e "\n${YELLOW}Searching for RTX 5090/4090 instances...${NC}"

# Search: RTX 5090 preferred, RTX 4090 fallback, >=24GB VRAM, cuda 12+
INSTANCE_ID=""
for gpu_name in "RTX_5090" "RTX_4090"; do
    echo "  Trying ${gpu_name}..."
    OFFERS=$(.venv/bin/vastai search offers \
        "gpu_name=${gpu_name} num_gpus=1 gpu_ram>=24 inet_down>=500 disk_space>=80 reliability>0.95" \
        --order "dph_total" --limit 3 --raw 2>/dev/null || true)

    if [ -n "$OFFERS" ] && [ "$OFFERS" != "[]" ]; then
        echo -e "  ${GREEN}Found ${gpu_name} offers${NC}"
        echo "$OFFERS" | head -5
        break
    fi
done

if [ -z "$OFFERS" ] || [ "$OFFERS" = "[]" ]; then
    echo -e "${RED}No suitable GPU instances found. Try again later.${NC}"
    exit 1
fi

# ---- Step 3: Create instance ----
echo -e "\n${YELLOW}Creating instance...${NC}"

# Use PyTorch nightly image with CUDA 12.8
OFFER_ID=$(echo "$OFFERS" | python3 -c "import sys,json; data=json.load(sys.stdin); print(data[0]['id'])" 2>/dev/null)

if [ -z "$OFFER_ID" ]; then
    echo -e "${RED}Failed to parse offer ID. Please create instance manually:${NC}"
    echo "  .venv/bin/vastai search offers 'gpu_name=RTX_4090 num_gpus=1 gpu_ram>=24'"
    echo "  .venv/bin/vastai create instance <OFFER_ID> --image pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel --disk 80"
    echo ""
    echo "Then run the remote script manually:"
    echo "  scp scripts/remote_generate.py <instance>:/workspace/"
    echo "  ssh <instance> 'cd /workspace && python remote_generate.py'"
    exit 1
fi

echo "  Offer ID: ${OFFER_ID}"
RESULT=$(.venv/bin/vastai create instance "${OFFER_ID}" \
    --image "pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel" \
    --disk 80 \
    --onstart-cmd "pip install diffusers transformers accelerate safetensors soundfile pillow" \
    --raw 2>/dev/null)

INSTANCE_ID=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('new_contract', ''))" 2>/dev/null)
echo -e "  ${GREEN}Instance created: ${INSTANCE_ID}${NC}"

# ---- Step 4: Wait for instance to start ----
echo -e "\n${YELLOW}Waiting for instance to start...${NC}"
for i in $(seq 1 60); do
    STATUS=$(.venv/bin/vastai show instance "${INSTANCE_ID}" --raw 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('actual_status', 'unknown'))" 2>/dev/null || echo "unknown")
    if [ "$STATUS" = "running" ]; then
        echo -e "  ${GREEN}Instance running!${NC}"
        break
    fi
    echo "  Status: ${STATUS} (${i}/60)..."
    sleep 10
done

# Get SSH details
SSH_CMD=$(.venv/bin/vastai ssh-url "${INSTANCE_ID}" 2>/dev/null)
echo "  SSH: ${SSH_CMD}"

# ---- Step 5: Upload files ----
echo -e "\n${YELLOW}Uploading code and narration audio...${NC}"

# Create tarball of necessary files
TARBALL="/tmp/animatediff_deploy.tar.gz"
cd "${PROJECT_ROOT}"
tar czf "${TARBALL}" \
    animatediff/ \
    scripts/produce_trailer_v2.py \
    scripts/generate_portraits.py \
    examples/fanren_trailer_v2.json \
    output/fanren-trailer-v2/narration/ \
    output/fanren-trailer-v2/storyboard_computed.json \
    output/fanren-trailer-v2/shot_timing.json \
    2>/dev/null

echo "  Tarball: $(du -h ${TARBALL} | cut -f1)"

# Upload via SCP (extract SSH host/port from SSH URL)
.venv/bin/vastai copy "${TARBALL}" "${INSTANCE_ID}:/workspace/deploy.tar.gz"
.venv/bin/vastai execute "${INSTANCE_ID}" "cd /workspace && tar xzf deploy.tar.gz && echo 'Upload complete'"

# ---- Step 6: Run generation ----
echo -e "\n${YELLOW}Running Phase 2 on GPU...${NC}"
.venv/bin/vastai execute "${INSTANCE_ID}" \
    "cd /workspace && python scripts/produce_trailer_v2.py --phase 2" 2>&1

# ---- Step 7: Download results ----
echo -e "\n${YELLOW}Downloading generated shots...${NC}"

.venv/bin/vastai execute "${INSTANCE_ID}" \
    "cd /workspace && tar czf /workspace/results.tar.gz output/fanren-trailer-v2/portraits/ output/fanren-trailer-v2/shots/"

.venv/bin/vastai copy "${INSTANCE_ID}:/workspace/results.tar.gz" "/tmp/results.tar.gz"
cd "${PROJECT_ROOT}" && tar xzf /tmp/results.tar.gz

echo -e "${GREEN}Downloaded portraits and shots to ${OUTPUT_DIR}${NC}"

# ---- Step 8: Destroy instance ----
echo -e "\n${YELLOW}Destroying instance to stop billing...${NC}"
.venv/bin/vastai destroy instance "${INSTANCE_ID}"
echo -e "${GREEN}Instance destroyed.${NC}"

# ---- Summary ----
echo -e "\n${GREEN}============================================${NC}"
echo -e "${GREEN} Phase 2 Complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "Portraits: ${OUTPUT_DIR}/portraits/"
echo "Shots:     ${OUTPUT_DIR}/shots/"
echo ""
echo "Next: Run Phase 3 locally:"
echo "  python scripts/produce_trailer_v2.py --phase 3"
