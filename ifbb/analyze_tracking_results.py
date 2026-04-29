import pandas as pd
import argparse
import os

def analyze_logs(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: Log file {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    
    print(f"--- Tracking Log Analysis: {csv_path} ---")
    print(f"Total entries: {len(df)}")
    print(f"Unique Track IDs: {df['track_id'].nunique()}")
    
    # Analyze each track
    for track_id in df['track_id'].unique():
        track_df = df[df['track_id'] == track_id]
        print(f"\nTrack ID {track_id}:")
        print(f"  Frames: {len(track_df)}")
        
        # Most frequent winner_match
        winner_counts = track_df['winner_match'].value_counts()
        if not winner_counts.empty:
            top_winner = winner_counts.index[0]
            top_count = winner_counts.iloc[0]
            print(f"  Primary Identity: {top_winner} ({top_count} frames)")
        
        # Best matches distribution
        best_match_counts = track_df['best_match'].value_counts().head(5)
        print("  Top 5 Best Matches (Raw Frame-by-Frame):")
        for name, count in best_match_counts.items():
            avg_sim = track_df[track_df['best_match'] == name]['best_sim'].mean()
            print(f"    - {name}: {count} frames (Avg Sim: {avg_sim:.4f})")
            
        # Margin analysis
        avg_margin = track_df['margin'].mean()
        print(f"  Average Confidence Margin (Best vs 2nd): {avg_margin:.4f}")
        
        if avg_margin < 0.05:
            print("  [WARNING] Low margin! The model is struggling to distinguish this track from other athletes.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to tracking debug log CSV")
    args = parser.parse_args()
    
    analyze_logs(args.csv)
