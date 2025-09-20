
import re
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

# Suppress openpyxl conditional formatting warnings that occur when reading Excel files
# These warnings are not critical for our data extraction process
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')
_HAS_DF_MAP = hasattr(pd.DataFrame, "map")

def _norm(s: str) -> str:
    """
    Normalize a string for matching purposes by:
    1. Converting to lowercase 
    2. Removing all non-alphanumeric characters (spaces, punctuation, etc.)
    This helps with fuzzy matching of test names and sample IDs that may have variations in formatting
    """
    # Handle None/null values by returning empty string
    if s is None:
        return ""
    # Convert to string (in case we get numeric values)
    s = str(s)
    # Convert to lowercase for case-insensitive matching
    s = s.lower()
    # Remove all characters that are not letters or numbers using regex
    return re.sub(r'[^a-z0-9]+', '', s)

def _make_test_keys(test: str):
    """
    Generate multiple possible variations of a test name to improve matching.
    This handles common variations in how lab tests are named across different sheets.
    For example: "Density" might appear as "dens", "sg", "specific gravity", etc.
    """
    # Normalize the input test name using the _norm function
    base = _norm(test)
    # Start with a set containing the normalized base name
    keys = {base}
    
    # Add specific variations for density-related tests
    if 'density' in base:
        keys.update({'density','dens','sg','specificgravity','densityat15c','densityat15','d15','api'})
    
    # Add specific variations for molecular weight tests
    if 'mw' in base or 'mwt' in base or 'molecular' in base:
        keys.update({'mwt','mw','molecularweight','avgmwt','avemw','averagemw','average_mw','avemwt'})
    
    # Create bidirectional replacements for common abbreviations
    repl = [('average','avg'), ('avg','average'), ('molecularweight','mw'), ('mwt','mw')]
    for a,b in repl:
        # If pattern 'a' is in base, add version with 'b' substituted
        if a in base: keys.add(base.replace(a,b))
        # If pattern 'b' is in base, add version with 'a' substituted  
        if b in base: keys.add(base.replace(b,a))
    
    # Remove temperature specifications (e.g., "at15c" from "densityat15c")
    keys.add(re.sub(r'at\d+c$', '', base))
    
    # Remove "average" and "avg" prefixes to create simpler versions
    keys.add(base.replace('average','').replace('avg',''))
    
    # Filter out empty strings that might result from replacements
    keys = {k for k in keys if k}
    return list(keys)

def _extract_first_float(val):
    """
    Extract the first numeric value from a cell that might contain mixed text and numbers.
    For example: "25.3 mg/L" -> 25.3, "Result: 0.85" -> 0.85
    This handles cases where test results are embedded in descriptive text.
    """
    # Handle None/null values
    if val is None:
        return None
    
    # Convert to string to handle numeric inputs as well as text
    s = str(val)
    
    # Use regex to find the first number (integer or decimal, possibly negative)
    # Pattern: optional minus sign, digits, optional decimal point and more digits
    m = re.search(r'(-?\d+(?:\.\d+)?)', s)
    
    # If no number pattern found, return None
    if not m:
        return None
    
    # Try to convert the matched string to float
    try:
        return float(m.group(1))
    except:
        # If conversion fails for any reason, return None
        return None

def extract_lab(File: str, sample_connection: str, test: str, debug: bool=False,
                header_cols: int = 60, scan_down: int = 60):
    """
    Enhanced extractor for complex lab sheets with multi-column samples and tests.
    
    This is the main function that extracts specific test results for a given sample
    from Excel lab reports that may have complex layouts with multiple time points.
    
    Parameters:
    - File: Path to the Excel file containing lab data
    - sample_connection: The sample ID to look for (e.g., "SC-101", "V-25")  
    - test: The exact test name to search for (e.g., "Density", "MW")
    - debug: If True, returns detailed information about where values were found
    - header_cols: Number of columns to scan for headers (unused in current implementation)
    - scan_down: Number of rows to scan down (unused in current implementation)
    
    Strategy:
      1) Find sample connections and map them to ALL their columns (ST1, ST2, ST3, ST4).
      2) Find test names in ANY column (not just left header area).
      3) Extract ALL values for the sample-test combination and average them.
      4) Return the average of collected values. If debug=True, include provenance.
    """
    # Import defaultdict for efficient data grouping (though not used in current implementation)
    from collections import defaultdict
    
    # Convert file path to Path object for easier manipulation
    p = Path(File)
    
    # If the file doesn't exist at the given path, try common alternative locations
    if not p.exists():
        # List of candidate paths where the file might be located
        candidates = [
            Path('/mnt/data')/File,  # Try in /mnt/data directory with full path
            Path('/mnt/data')/Path(File).name,  # Try in /mnt/data with just filename
            Path('/mnt/data/13.08.2025.xlsx') if '13.08.2025' in File else None,  # Specific fallback file
        ]
        # Filter out None values from candidates list
        candidates = [c for c in candidates if c is not None]
        
        # Try each candidate location until we find an existing file
        for c in candidates:
            if c.exists():
                p = c
                break
        else:
            # If none of the candidates exist, raise an error
            raise FileNotFoundError(f"Cannot find Excel file: {File}")

    # Normalize the sample connection name for matching using _norm function
    sample_key = _norm(sample_connection)
    # Create an exact normalized version of the test name for precise matching
    test_key_exact = _norm(test)  # Exact match for the test name
    # Create a base normalized version for bounds checking later
    base = _norm(test)

    # Load the Excel file using pandas ExcelFile for multi-sheet access
    xls = pd.ExcelFile(p)
    # Initialize list to collect all matching values found across sheets
    hits = []

    # Iterate through each sheet in the Excel workbook
    for sheet in xls.sheet_names:
        try:
            # Parse the sheet without assuming any header structure (header=None)
            # Use dtype=object to preserve all data types and avoid unwanted conversions
            df = xls.parse(sheet_name=sheet, header=None, dtype=object)
        except Exception:
            # If sheet parsing fails, skip to next sheet
            continue
        
        # Get the dimensions of the current sheet
        rows, cols = df.shape
        
        # Create a normalized version of the dataframe for matching purposes
        # Apply the _norm function to every cell in the dataframe

        df_norm = (df.map if _HAS_DF_MAP else df.applymap)(_norm)

        # STEP 1: Find the exact sample and identify all its associated columns
        target_sample_columns = set()  # Will store column indices for our specific target sample
        sample_section_row = None  # Track which row our sample is found in (for section context)
        
        # Scan through every cell in the sheet to find sample connections
        for row_idx in range(rows):
            for col_idx in range(cols):
                # Get both normalized and original cell values
                cell_norm = df_norm.iat[row_idx, col_idx] or ''  # Normalized for matching
                cell_orig = str(df.iat[row_idx, col_idx] or '').strip()  # Original for display
                
                # Check if this cell contains our target sample (using normalized comparison)
                if sample_key and sample_key in cell_norm:
                    # Mark this column as belonging to our target sample
                    target_sample_columns.add(col_idx)
                    # Remember which row we found the sample in (used for limiting search scope)
                    sample_section_row = row_idx
                    
                    # STEP 1A: Find additional columns belonging to this sample
                    # Look for the boundary where the next sample begins
                    next_sample_col = cols  # Default to end of sheet if no next sample found
                    
                    # Scan columns to the right to find where the next sample starts
                    for next_col in range(col_idx + 1, cols):
                        next_cell = str(df.iat[row_idx, next_col] or '').strip()
                        # Look for common sample ID patterns that indicate a new sample
                        if next_cell and ('SC-' in next_cell or 'V-' in next_cell or 'SAM' in next_cell):
                            next_sample_col = next_col
                            break
                    
                    # STEP 1B: Find time-point columns (ST1, ST2, ST3, ST4) for this sample
                    time_columns = set()
                    # Check the next few rows below the sample header for time indicators
                    for time_row_offset in range(1, 4):  # Check rows +1, +2, +3 from sample row
                        time_row_idx = row_idx + time_row_offset
                        if time_row_idx < rows:  # Make sure we don't go beyond sheet boundaries
                            # Only check columns between our sample and the next sample
                            for check_col in range(col_idx + 1, next_sample_col):
                                time_val = str(df.iat[time_row_idx, check_col] or '').strip()
                                # Look for time indicators: ST1/ST2/ST3/ST4 or time format (HH:MM)
                                if (time_val in ['ST1', 'ST2', 'ST3', 'ST4'] or 
                                    (':' in time_val and len(time_val) >= 4)):  # More specific time pattern detection
                                    time_columns.add(check_col)
                    
                    # STEP 1C: Add the time columns to our target sample columns
                    # If no next sample found, limit to the actual time columns found
                    if next_sample_col == cols and time_columns:
                        # Use only the time columns we actually found (conservative approach)
                        target_sample_columns.update(time_columns)
                    else:
                        # Use the time columns within the boundary if we have a clear next sample boundary
                        target_sample_columns.update(time_columns)

        # STEP 2: Find test rows that match our exact test name within the sample's section
        test_rows = set()  # Will store row indices where our test is found
        
        # Only search if we actually found our sample in this sheet
        if sample_section_row is not None:
            # Define search boundaries around where our sample was found to avoid false matches
            search_start = max(0, sample_section_row - 5)  # Look a few rows above sample
            search_end = min(rows, sample_section_row + 50)  # Look reasonable distance below sample
            
            # Search for the exact test name within the defined boundaries
            for test_row in range(search_start, search_end):
                for test_col in range(cols):  # Search ALL columns for test names (not just leftmost)
                    cell_orig = str(df.iat[test_row, test_col] or '').strip()
                    # Use exact string matching on original text (not normalized) 
                    # This preserves important characters like "=" that might be part of test names
                    if cell_orig == test:
                        test_rows.add(test_row)

        # STEP 3: Extract values from the intersection of test rows and target sample columns
        # Only proceed if we found both the sample columns AND the test rows
        if target_sample_columns and test_rows:
            # For each test row that contains our test name
            for test_row in test_rows:
                # For each column belonging to our target sample
                for data_col in target_sample_columns:
                    # Try to extract a numeric value using the _extract_first_float function
                    val = _extract_first_float(df.iat[test_row, data_col])
                    
                    # If no numeric value was extracted, handle special text cases
                    if val is None:
                        cell_val = str(df.iat[test_row, data_col] or '').strip().upper()
                        
                        # Handle "NIL" as zero (common in lab reports for non-detect)
                        if cell_val == 'NIL':
                            val = 0.0
                        # Handle less-than values (e.g., "<0.5" becomes 0.5)
                        elif cell_val.startswith('<'):
                            try:
                                val = float(cell_val[1:])  # Remove '<' and convert to float
                            except:
                                continue  # Skip if conversion fails
                        else:
                            continue  # Skip cells that don't contain usable values
                    
                    # STEP 3A: Apply reasonable bounds checking to filter out clearly erroneous values
                    ok = True  # Assume value is OK by default
                    
                    # Apply density-specific bounds (typical range for petroleum products)
                    if 'density' in base:
                        ok = 200 <= val <= 2000  # Reasonable density range in kg/m³
                    # Apply molecular weight bounds (typical range for organic compounds)
                    elif any(k in base for k in ['mw','mwt','molecular']):
                        ok = 2 <= val <= 200  # Reasonable molecular weight range
                    
                    # Only add the value to hits if it passes bounds checking
                    if ok:
                        # Store the value along with its provenance information for debugging
                        # Convert to 1-based indexing for user-friendly Excel coordinates
                        hits.append((float(val), sheet, test_row+1, data_col+1, 
                                   f"from sample {sample_connection} col {data_col+1} at test row {test_row+1}"))

    # STEP 4: Handle case where no matching values were found
    if not hits:
        # Return appropriate response based on debug mode
        return {"value": None} if not debug else {"value": None, "debug": [], "reason":"No values found."}

    # STEP 5: Remove duplicate values that come from the same cell location
    values = []  # List of unique numeric values
    prov = []    # List of provenance information for debugging
    seen = set() # Set to track already processed locations
    
    # Process each hit and remove duplicates
    for val, sheet, r, c, why in hits:
        # Create a unique key for this cell location and value (rounded to avoid floating point issues)
        key = (sheet, r, c, round(val, 6))
        
        # Skip if we've already processed this exact location and value
        if key in seen:
            continue
        
        # Mark this location/value as seen
        seen.add(key)
        # Add to our list of values to average
        values.append(val)
        
        # If debug mode, collect provenance information
        if debug:
            prov.append({"file": p.name, "sheet": sheet, "excel_row": r, "excel_col": c, "value": val, "why": why})

    # STEP 6: Calculate the final result by averaging all found values
    avg = float(np.mean(values))  # Use numpy.mean for reliable averaging
    
    # Return results based on debug mode
    if debug:
        # Debug mode: return comprehensive information about the extraction
        return {
            "value": avg,                    # The final averaged value
            "count": len(values),           # How many values were averaged
            "values_used": values,          # List of individual values that were averaged
            "provenance": prov[:2000]       # Detailed information about where each value came from (limited to 2000 entries)
        }
    else:
        # Normal mode: return just the averaged value
        return avg
