"""
chunking.py
Extracts structured chunks from PDF technical data sheets.

Strategy:
- Uses PyMuPDF (fitz) to parse PDFs page by page
- Separates TEXT blocks from TABLE blocks using bounding box intersection
- Detects section headings by font size (>= 11.5pt, < 60 chars)
- Tables are classified as REGULATORY (food safety limits) or GENERIC
- Regulatory tables → one chunk per row (e.g., "Mercury: <0.5 mg/kg")
- Generic tables → header:value pairs per row (e.g., "Dosage: 5-30 ppm")
- Text blocks → grouped under their parent section heading
- Identical chunks from multiple PDFs are MERGED, accumulating ingredient names

This produces a flat JSON of chunks, each with:
  - update date, ingredients[], section title, chunk text OR table row data
"""
import fitz
import json
import os
import glob
import re

def process_pdfs(folder_path):
    all_chunks = []
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    
    for pdf_path in pdf_files:
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"Failed to open {pdf_path}: {e}")
            continue
            
        file_name = os.path.basename(pdf_path)
        ingredient_name = file_name.replace(".pdf", "").strip()

        # Extract Update Date globally
        update_date = ""
        for page in doc:
            text = page.get_text()
            match = re.search(r"Last updating:\s*([\d/]+)", text, re.IGNORECASE)
            if match:
                update_date = match.group(1).strip()
                break

        current_section = "General"
        current_text = []

        def save_text_chunk():
            nonlocal current_section, current_text
            if current_section != "General" or current_text:
                joined_text = current_section + ("\n" + "\n".join(current_text) if current_text else "")
                all_chunks.append({
                    "update date": update_date,
                    "ingredient": ingredient_name,
                    "section title": current_section,
                    "chunk text": joined_text.strip()
                })
            current_text = []

        for page in doc:
            items_on_page = []
            
            # 1. Grab tables from PyMuPDF
            tables = list(page.find_tables())
            table_bboxes = []
            for t_idx, t in enumerate(tables):
                t_rect = fitz.Rect(t.bbox)
                table_bboxes.append(t_rect)
                # Keep y0 quantized for sorting
                items_on_page.append({
                    "type": "table",
                    "sort_key": (round(t_rect.y0/10)*10, t_rect.x0),
                    "data": t
                })
            
            # 2. Grab text blocks
            dict_output = page.get_text("dict")
            blocks = [b for b in dict_output.get("blocks", []) if b["type"] == 0]
            
            for b in blocks:
                b_rect = fitz.Rect(b["bbox"])
                
                # Check intersection with ANY table
                # We do a slight shrink to avoid border false positive text inclusions
                reduced_rect = b_rect + (1, 1, -1, -1) 
                
                in_table = False
                for tbox in table_bboxes:
                    if reduced_rect.intersects(tbox):
                        in_table = True
                        break
                        
                if not in_table:
                    items_on_page.append({
                        "type": "text",
                        "sort_key": (round(b_rect.y0/10)*10, b_rect.x0),
                        "data": b
                    })
                    
            # 3. Process chronologically top-to-bottom
            items_on_page.sort(key=lambda x: x["sort_key"])
            
            last_category = "" # persisting across a table

            for item in items_on_page:
                if item["type"] == "text":
                    b = item["data"]
                    max_size = 0
                    spans_texts = []
                    for l in b["lines"]:
                        for s in l["spans"]:
                            if s["size"] > max_size: max_size = s["size"]
                            t = s["text"].strip()
                            if t: spans_texts.append(t)
                    
                    if not spans_texts: continue
                    full_text = " ".join(spans_texts)
                    clean_text = full_text.lower()
                    
                    if ("vtr&beyond" in clean_text or "www.vtrbeyond.com" in clean_text or 
                        "info@" in clean_text or "stresemann str" in clean_text or 
                        "technical data sheet" in clean_text or full_text.endswith("®") or 
                        "last updating:" in clean_text):
                        continue
                    if "bakery enzyme" in clean_text and len(clean_text) < 20: continue

                    is_heading = max_size >= 11.5 and len(full_text) < 60
                    if is_heading:
                        save_text_chunk()
                        current_section = full_text
                    else:
                        current_text.append(full_text)
                        
                elif item["type"] == "table":
                    t = item["data"]
                    save_text_chunk() # Table breaks paragraph flow
                    t_df = t.extract()
                    if not t_df: continue
                    
                    # Detect if it's Regulatory Limits or Generic
                    is_regulatory = "FOOD SAFTY" in current_section.upper()
                    if not is_regulatory and t_df and t_df[0]:
                        header_str = " ".join([str(x).upper() for x in t_df[0] if x])
                        if "FOOD SAFTY" in header_str:
                            is_regulatory = True
                            
                    if is_regulatory:
                        table_title = "FOOD SAFTY DATA" if current_section == "General" else current_section
                        for row in t_df:
                            cells = [str(c).replace('\n', ' ').strip() if c else "" for c in row]
                            if not cells or all(not c for c in cells): continue
                            
                            joined_row = " ".join(cells).upper()
                            if "FOOD SAFTY" in joined_row:
                                continue
                                
                            leftmost_idx = next((i for i, c in enumerate(cells) if c), -1)
                            rightmost_idx = next((i for i in range(len(cells)-1, -1, -1) if cells[i]), -1)
                            
                            if leftmost_idx == -1: continue
                            left_val = cells[leftmost_idx]
                            right_val = cells[rightmost_idx]
                            
                            if leftmost_idx == rightmost_idx and ":" not in left_val and len(left_val) < 40:
                                last_category = left_val
                                continue
                                
                            if leftmost_idx < rightmost_idx:
                                if left_val and left_val != right_val:
                                    last_category = left_val
                                row_text = right_val
                            else:
                                row_text = left_val
                                
                            if not last_category:
                                last_category = "General"
                                
                            all_chunks.append({
                                "update date": update_date,
                                "ingredient": ingredient_name,
                                "section title": table_title,
                                "table type": "Regulatory limits",
                                "row title": last_category,
                                "row text": row_text
                            })
                    else:
                        # Generic Table Processing
                        # Use first row as headers
                        headers = [str(c).replace('\n', ' ').strip() if c else "" for c in t_df[0]]
                        cleaned_headers = [h if h else f"Column {i+1}" for i, h in enumerate(headers)]
                        
                        all_empty = True
                        for h in cleaned_headers:
                            if not h.startswith("Column "): all_empty = False
                        
                        start_idx = 1
                        if all_empty: 
                            # If row 0 is completely empty, start from 0 and just use generic columns
                            start_idx = 0
                            
                        for r_idx in range(start_idx, len(t_df)):
                            cells = [str(c).replace('\n', ' ').strip() if c else "" for c in t_df[r_idx]]
                            if not cells or all(not c for c in cells): continue
                            
                            first_val = next((c for c in cells if c), f"Row {r_idx}")
                            
                            parts = []
                            for i, c in enumerate(cells):
                                if c:
                                    hdr = cleaned_headers[i] if i < len(cleaned_headers) else f"Column {i+1}"
                                    parts.append(f"{hdr}: {c}")
                            
                            row_text = " | ".join(parts)
                            
                            all_chunks.append({
                                "update date": update_date,
                                "ingredient": ingredient_name,
                                "section title": current_section,
                                "table type": "Generic Table",
                                "row title": first_val,
                                "row text": row_text
                            })

            save_text_chunk() # End of page texts if any

    return all_chunks

def merge_identical_chunks(chunks):
    merged_dict = {}
    for c in chunks:
        if "table type" in c:
            key = (
                c.get("update date", ""),
                c.get("section title", ""),
                c.get("table type", ""),
                c.get("row title", ""),
                c.get("row text", "")
            )
            is_table = True
        else:
            key = (
                c.get("update date", ""),
                c.get("section title", ""),
                c.get("chunk text", "")
            )
            is_table = False
            
        if key not in merged_dict:
            if is_table:
                merged_dict[key] = {
                    "update date": c.get("update date", ""),
                    "ingredients": [c["ingredient"]],
                    "section title": c.get("section title", ""),
                    "table type": c.get("table type", ""),
                    "row title": c.get("row title", ""),
                    "row text": c.get("row text", "")
                }
            else:
                merged_dict[key] = {
                    "update date": c.get("update date", ""),
                    "ingredients": [c["ingredient"]],
                    "section title": c.get("section title", ""),
                    "chunk text": c.get("chunk text", "")
                }
        else:
            if c["ingredient"] not in merged_dict[key]["ingredients"]:
                merged_dict[key]["ingredients"].append(c["ingredient"])
                
    return list(merged_dict.values())

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(script_dir, "enzymes")
    output_file = os.path.join(script_dir, "all_extracted_chunks_merged.json")
    
    print("Process starting...")
    raw_chunks = process_pdfs(folder)
    print(f"Extracted {len(raw_chunks)} raw chunks.")
    
    final_merged_chunks = merge_identical_chunks(raw_chunks)
    print(f"Compressed down to {len(final_merged_chunks)} unique merged chunks.")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_merged_chunks, f, indent=2, ensure_ascii=False)
        
    print(f"Output chunks successfully saved to {output_file}")
