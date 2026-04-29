import os
import sqlite3
import torch
import zipfile
import shutil
import pytest
from pathlib import Path
from ifbb.prepare_bodybuilding_dataset import BodybuildingDatasetBuilder
from ifbb.check_db_distribution import check_distribution

@pytest.fixture
def mock_npc_db(tmp_path):
    """Creates a mock NPC database for testing."""
    db_path = tmp_path / "npc_database.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE athletes (
            id INTEGER PRIMARY KEY,
            year INTEGER,
            contest_name TEXT,
            division TEXT,
            placing INTEGER,
            athlete_name TEXT,
            image_filename TEXT
        )
    """)
    
    # Add dummy data
    data = [
        (1, 2025, "Prague Pro", "MEN'S BODYBUILDING", 1, "Martin Fitzwater", "fitz_1.jpg"),
        (2, 2025, "Prague Pro", "MEN'S BODYBUILDING", 2, "Samson Dauda", "dauda_1.jpg"),
        (3, 2025, "Prague Pro", "MEN'S PHYSIQUE", 1, "Youssef Cherrate", "youssef_1.jpg"),
        (4, 2024, "Olympia", "212", 1, "Keone Pearson", "keone_1.jpg")
    ]
    cursor.executemany("INSERT INTO athletes VALUES (?, ?, ?, ?, ?, ?, ?)", data)
    conn.commit()
    conn.close()
    return db_path

@pytest.fixture
def mock_source_dir(tmp_path):
    """Creates a mock source directory with contest zips."""
    source_root = tmp_path / "source"
    source_root.mkdir()
    
    # Create 2025 folder and zip
    dir_2025 = source_root / "2025"
    dir_2025.mkdir()
    
    # Create temp files to zip (actual valid JPGs)
    import cv2
    import numpy as np
    temp_files = tmp_path / "temp_files"
    temp_files.mkdir()
    
    def save_dummy_img(path):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(path), img)

    save_dummy_img(temp_files / "fitz_1.jpg")
    save_dummy_img(temp_files / "dauda_1.jpg")
    save_dummy_img(temp_files / "youssef_1.jpg")
    save_dummy_img(temp_files / "keone_1.jpg")
    save_dummy_img(temp_files / "turek_1.jpg")

    zip_2025 = dir_2025 / "2025_Prague_Pro.zip"
    with zipfile.ZipFile(zip_2025, 'w') as z:
        z.write(temp_files / "fitz_1.jpg", arcname="fitz_1.jpg")
        z.write(temp_files / "dauda_1.jpg", arcname="dauda_1.jpg")
        z.write(temp_files / "youssef_1.jpg", arcname="youssef_1.jpg")
        
    # Create 2024 folder and zip
    dir_2024 = source_root / "2024"
    dir_2024.mkdir()
    
    zip_2024 = dir_2024 / "2024_Olympia.zip"
    with zipfile.ZipFile(zip_2024, 'w') as z:
        z.write(temp_files / "keone_1.jpg", arcname="keone_1.jpg")
        
    return source_root

from unittest.mock import MagicMock

def test_dataset_builder_filtering(mock_npc_db, mock_source_dir, tmp_path):
    """Test that BodybuildingDatasetBuilder correctly filters by division."""
    output_dir = tmp_path / "output"
    
    builder = BodybuildingDatasetBuilder(
        source_root=mock_source_dir,
        db_path=mock_npc_db,
        output_dir=output_dir,
        device="cpu"
    )
    
    # Mock the detector to return a fake result with a box
    mock_box = MagicMock()
    mock_box.cls = torch.tensor([0])
    mock_box.conf = torch.tensor([0.9])
    mock_box.xywhn = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    
    mock_result = MagicMock()
    mock_result.boxes = [mock_box]
    
    builder.detector = MagicMock(return_value=[mock_result, mock_result])
    
    # Filter for Bodybuilding only
    builder.build(target_divisions=["MEN'S BODYBUILDING"])
    
    img_dir = output_dir / "images"
    lbl_dir = output_dir / "labels"
    
    # Should have Fitzwater and Dauda, but NOT Youssef (Physique) or Keone (212)
    assert len(list(img_dir.glob("*.jpg"))) == 2
    assert len(list(lbl_dir.glob("*.txt"))) == 2
    
    # Check if IDs are consistent in mapping DB
    conn = sqlite3.connect(output_dir / "dataset_id_mapping.db")
    cursor = conn.cursor()
    cursor.execute("SELECT athlete_name FROM id_mapping")
    athletes = [r[0] for r in cursor.fetchall()]
    assert "MARTIN FITZWATER" in athletes
    assert "SAMSON DAUDA" in athletes
    assert "YOUSSEF CHERRATE" not in athletes
    conn.close()

def test_dataset_builder_resuming(mock_npc_db, mock_source_dir, tmp_path):
    """Test that the builder correctly resumes (skips processed contests)."""
    output_dir = tmp_path / "output_resume"
    builder = BodybuildingDatasetBuilder(
        source_root=mock_source_dir,
        db_path=mock_npc_db,
        output_dir=output_dir,
        device="cpu"
    )

    # Mock the detector to return a fake result with a box
    mock_box = MagicMock()
    mock_box.cls = torch.tensor([0])
    mock_box.conf = torch.tensor([0.9])
    mock_box.xywhn = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    mock_result = MagicMock()
    mock_result.boxes = [mock_box]
    builder.detector = MagicMock(return_value=[mock_result, mock_result])
    
    # First run
    builder.build(target_divisions=["MEN'S BODYBUILDING"])
    assert builder.is_contest_processed(2025, "Prague Pro")
    
    # Modify mock DB to add a new contest
    conn = sqlite3.connect(mock_npc_db)
    conn.execute("INSERT INTO athletes VALUES (5, 2024, 'New Contest', 'MEN''S BODYBUILDING', 1, 'Jan Turek', 'turek_1.jpg')")
    conn.commit()
    conn.close()
    
    # Create zip for new contest
    dir_2024 = mock_source_dir / "2024"
    (tmp_path / "turek_1.jpg").touch()
    with zipfile.ZipFile(dir_2024 / "2024_New_Contest.zip", 'w') as z:
        z.write(tmp_path / "turek_1.jpg", arcname="turek_1.jpg")
        
    # Second run - should only process New Contest
    # We can check this by seeing if the count of images in output folder increases
    builder.build(target_divisions=["MEN'S BODYBUILDING"])
    img_dir = output_dir / "images"
    # 2 from first run + 1 from second run
    assert len(list(img_dir.glob("*.jpg"))) == 3

def test_check_distribution(mock_npc_db, capsys):
    """Test that check_distribution script prints correctly."""
    check_distribution(str(mock_npc_db))
    captured = capsys.readouterr()
    assert "MEN'S BODYBUILDING" in captured.out
    assert "MEN'S PHYSIQUE" in captured.out
    assert "Prague Pro" in captured.out
