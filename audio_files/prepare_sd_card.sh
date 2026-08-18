#!/bin/bash
# Generates voice clips as MP3 and copies them to the DFPlayer's SD card.
#
# Requires: ffmpeg (install with: brew install ffmpeg)
#
# Usage:
#   1. Format the SD card as FAT via Disk Utility
#   2. Run:  bash prepare_sd_card.sh /Volumes/CANE

set -e

CARD="$1"

if [ -z "$CARD" ] || [ ! -d "$CARD" ]; then
    echo "Usage: bash prepare_sd_card.sh /Volumes/YOUR_CARD_NAME"
    exit 1
fi

if ! command -v ffmpeg &>/dev/null; then
    echo "ffmpeg not found. Install with:  brew install ffmpeg"
    exit 1
fi

echo "Generating clips..."
TMPDIR=$(mktemp -d)

say -o "$TMPDIR/temp1.aiff" "obstacle left"
say -o "$TMPDIR/temp2.aiff" "obstacle right"
say -o "$TMPDIR/temp3.aiff" "obstacle ahead"

echo "Converting to MP3..."
ffmpeg -i "$TMPDIR/temp1.aiff" -b:a 128k "$TMPDIR/0001.mp3" -y -loglevel quiet
ffmpeg -i "$TMPDIR/temp2.aiff" -b:a 128k "$TMPDIR/0002.mp3" -y -loglevel quiet
ffmpeg -i "$TMPDIR/temp3.aiff" -b:a 128k "$TMPDIR/0003.mp3" -y -loglevel quiet

echo "Copying to $CARD in order..."
cp "$TMPDIR/0001.mp3" "$CARD/"
cp "$TMPDIR/0002.mp3" "$CARD/"
cp "$TMPDIR/0003.mp3" "$CARD/"

echo "Cleaning macOS metadata..."
dot_clean "$CARD"
find "$CARD" -name "._*" -delete 2>/dev/null || true

echo "Files on card:"
ls -la "$CARD"/*.mp3

echo ""
echo "Done. Eject:  diskutil eject $CARD"

rm -rf "$TMPDIR"
