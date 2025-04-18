#!/bin/bash

# Create output directory if it doesn't exist
mkdir -p /home/apo/Code/tmp/cvp_lect/notes_md

# Change to the directory containing markitdown
cd /home/apo/Code/Utils/markitdown

# Loop through all PDF files in Notlar directory
for pdf_file in /home/apo/Code/tmp/cvp_lect/Notlar/*.pdf; do
    # Get the filename without path and extension
    filename=$(basename "$pdf_file" .pdf)
    
    # Convert PDF to Markdown using markitdown from current directory
    markitdown "$pdf_file" > "/home/apo/Code/tmp/cvp_lect/notes_md/${filename}.md"
    
    echo "Converted $filename.pdf to ${filename}.md"
done

echo "All PDF files have been converted to Markdown"