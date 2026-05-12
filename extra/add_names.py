import pandas as pd

# ── CONFIGURATION ──────────────────────────────────────────────────────────
RAW_MATRIX_PATH = 'data/FW_005.csv'          # The one with just numbers
SPECIES_INFO_PATH = 'data/FW_005-species.csv' # The one with "Specie", "Role", etc.
OUTPUT_PATH = 'data/FW_005_NAMED.csv'        # The new file it will create

def rebuild_rectangular_matrix():
    # 1. Load the species info
    attr_df = pd.read_csv(SPECIES_INFO_PATH)
    
    # 2. Extract Prey names and Predator names based on the "Role" column
    # We use .unique() to ensure we don't have duplicate headers
    prey_names = attr_df[attr_df['Role'] == 'Prey']['Specie'].astype(str).unique().tolist()
    pred_names = attr_df[attr_df['Role'] == 'Predator']['Specie'].astype(str).unique().tolist()
    
    # 3. Load the numerical data
    # We read it raw (no headers) because we are going to apply our own
    matrix_data = pd.read_csv(RAW_MATRIX_PATH, header=None)
    
    rows, cols = matrix_data.shape
    print(f"Matrix detected: {rows} rows (Prey) x {cols} columns (Predators)")
    print(f"Species list has: {len(prey_names)} Prey and {len(pred_names)} Predators")

    # 4. Check for dimension mismatch
    if rows != len(prey_names) or cols != len(pred_names):
        print("\n!!! DIMENSION MISMATCH !!!")
        print(f"The CSV has {rows} rows, but your species list has {len(prey_names)} prey.")
        print("Trimming/Padding names to fit the data dimensions...")
        # Adjust name lists to match actual data shape
        prey_names = prey_names[:rows] if len(prey_names) > rows else prey_names + [f"Unknown_Prey_{i}" for i in range(rows-len(prey_names))]
        pred_names = pred_names[:cols] if len(pred_names) > cols else pred_names + [f"Unknown_Pred_{i}" for i in range(cols-len(pred_names))]

    # 5. Create the named DataFrame
    named_df = pd.DataFrame(
        data=matrix_data.values,
        index=prey_names,   # Rows = Prey
        columns=pred_names  # Columns = Predators
    )

    # 6. Save to CSV
    named_df.to_csv(OUTPUT_PATH, index_label='')
    
    print(f"\nSuccessfully created: {OUTPUT_PATH}")
    print(f"Shape: {named_df.shape}")

if __name__ == "__main__":
    rebuild_rectangular_matrix()